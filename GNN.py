import dgl
import torch
from torch import nn
import math
from dgl.nn import EdgeGATConv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_time_matrix(jobList, numberOfJobs, W, P, n):
    """构建Time矩阵，下半部分全为0"""
    time_matrix = torch.zeros((numberOfJobs, numberOfJobs))
    time_list = torch.zeros((1, numberOfJobs))

    for j in range(numberOfJobs):
        R_OMA = W * math.log(1 + jobList[j][0] * P / n)
        time_OMA = jobList[j][1] / R_OMA
        time_matrix[j, j] = time_OMA
        time_list[0, j] = time_OMA
        for k in range(j + 1, numberOfJobs):
            R_NOMA = W * math.log(1 + jobList[j][0] * P / (jobList[k][0] * P + n))
            time_NOMA = jobList[k][1] / R_NOMA
            time_matrix[j, k] = max(time_NOMA, time_OMA)

    return time_matrix, time_list


def tensor_to_dgl(tensor):  # 输入进来的tensor是增加terminal并且下方补0，处理成方阵的Graph
    graph = dgl.DGLGraph()
    num_nodes = tensor.shape[0]
    graph.add_nodes(num_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if tensor[i, j] == 1:
                graph.add_edges(i, j)

    return graph


class GraphNN(nn.Module):
    def __init__(self, output_features_number=6):
        super(GraphNN, self).__init__()
        self.output_features_number = output_features_number

        num_heads = 3
        self.conv1 = EdgeGATConv(in_feats=6, edge_feats=1, out_feats=8, num_heads=num_heads, allow_zero_in_degree=True).to(device)
        self.conv2 = EdgeGATConv(in_feats=8, edge_feats=1, out_feats=16, num_heads=num_heads, allow_zero_in_degree=True).to(device)
        self.conv3 = EdgeGATConv(in_feats=16, edge_feats=1, out_feats=32, num_heads=num_heads, allow_zero_in_degree=True).to(device)
        self.linear1 = nn.Sequential(nn.Linear(32, 16),
                                     nn.LeakyReLU(),
                                     nn.Linear(16, 4),
                                     nn.LeakyReLU(),
                                     nn.Linear(4, 1)).to(device)
        self.linear2 = nn.Linear(num_heads, 1).to(device)


    def forward(self, Graph, h, L, W, P, N):
        if isinstance(h, list):
            numberOfJobs = len(h)
            jobs = list(zip(h, L))
        elif isinstance(h, torch.Tensor):
            numberOfJobs = h.numel()
            jobs = list(zip(h.flatten().tolist(), L.flatten().tolist()))
            W, P, N = W.item(), P.item(), N.item()
        else:
            raise TypeError("What is h???")

        numberOfMachines = Graph.shape[1] - numberOfJobs
        jobList = torch.tensor(sorted(jobs, key=lambda x: x[0], reverse=True)).to(device)
        otherFeatures = torch.tensor([W, P, N, 1]).to(device)  # 1代表这个feature属于machine


        # jobFeatures形状为 n*6(h_i,L_i,W_i,P_i,n_i,1) 1代表这个feature属于machine
        jobFeatures = torch.cat((jobList, otherFeatures.unsqueeze(1).t().expand(jobList.size(0), -1)), dim=1)

        # machineFeatures填充全为2，大小为numberOfMachines*6
        machineFeatures = (torch.ones((numberOfMachines, 6)) * 2).to(device)

        # terminalFeatures填充全为3，大小为1*6
        terminalFeature = (torch.ones((1, 6)) * 3).to(device)

        node_features = torch.cat((jobFeatures, machineFeatures, terminalFeature), dim=0)


        # 在Graph右侧增加一列0，代表terminal
        add_terminal_tensor = torch.zeros(numberOfJobs, 1).to(device)
        Graph_with_terminal = torch.cat((Graph, add_terminal_tensor), dim=1)

        # 下方把machine连terminal的位置填1，其余填0组成方阵
        numberOfJMT = Graph_with_terminal.shape[1]
        tmp_tensor_left = torch.zeros((numberOfJMT - numberOfJobs, numberOfJMT - 1)).to(device)
        tmp_tensor_right = torch.ones((numberOfJMT - numberOfJobs, 1)).to(device)
        tmp_tensor_right[-1][-1] = 0
        add_zero_tensor = torch.cat((tmp_tensor_left, tmp_tensor_right), dim=1)
        Graph_edited = torch.cat((Graph_with_terminal, add_zero_tensor), dim=0)

        dgl_Graph = tensor_to_dgl(Graph_edited).to(device)


        # job与job边的feature为T矩阵对应的时间，其余边的feature都为0
        numberOfEdges = dgl_Graph.num_edges()
        time_matrix = build_time_matrix(jobList, numberOfJobs, W, P, N)[0].to(device)
        small_Graph = Graph[:, :numberOfJobs].to(device)
        indices = torch.nonzero(small_Graph == 1).to(device)
        extracted_elements = time_matrix[indices[:, 0], indices[:, 1]]
        m_to_m_edge_features = extracted_elements.view(-1, 1)
        other_edge_features = torch.zeros(numberOfEdges - m_to_m_edge_features.shape[0], 1).to(device)
        edge_features = torch.cat((m_to_m_edge_features, other_edge_features), dim=0)


        fst_time_node_feats = self.conv1(dgl_Graph, node_features, edge_features)
        snd_time_node_feats = self.conv2(dgl_Graph, fst_time_node_feats.mean(dim=1), edge_features)
        trd_time_node_feats = self.conv3(dgl_Graph, snd_time_node_feats.mean(dim=1), edge_features)

        feats = self.linear1(trd_time_node_feats).view(numberOfJMT, -1)
        feats = self.linear2(feats)
        feat = torch.max(feats, dim=0).values

        return feat


    def mask(self, Graph: torch.tensor):
        left = torch.ones((Graph.size()[0], Graph.size()[0])).to(device)
        right = torch.ones((Graph.size()[0], Graph.size()[1] - Graph.size()[0])).to(device)
        row = torch.sum(Graph, dim=1, keepdim=True)
        col = torch.sum(Graph, dim=0, keepdim=True)

        left = left - row - torch.t(row) - col[:, 0:Graph.size()[0]] - torch.t(col[:, 0:Graph.size()[0]])
        left = torch.where(left == 1, torch.tensor(1).float(), torch.tensor(0).float())
        left = left.triu(diagonal=1)
        right -= row

        return torch.cat((left, right), dim=1)



if __name__ == "__main__":
    dd = GraphNN()
    obs = {'Graph': torch.tensor([[0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0.]], device='cuda:0'), 'h': torch.tensor([9.4304e-12, 7.3497e-12, 7.3886e-12, 2.2969e-12], device='cuda:0'), 'L': torch.tensor([506, 427, 102, 973], device='cuda:0'), 'W': torch.tensor(90000., device='cuda:0'), 'P': torch.tensor(0.1000, device='cuda:0'), 'N': torch.tensor(3.5830e-16, device='cuda:0')}

    ret = dd.forward(obs['Graph'], obs['h'], obs['L'], obs['W'], obs['P'], obs['N'])

    print(dd.mask(obs['Graph']))

    print(ret)
