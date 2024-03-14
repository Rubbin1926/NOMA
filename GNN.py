import dgl
import torch
from torch import nn
from torch.nn import functional as F
import math
from dgl.nn import EdgeGATConv


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
        self.conv1 = EdgeGATConv(in_feats=6, edge_feats=1, out_feats=8, num_heads=num_heads, allow_zero_in_degree=True)
        self.conv2 = EdgeGATConv(in_feats=8, edge_feats=1, out_feats=16, num_heads=num_heads, allow_zero_in_degree=True)
        self.conv3 = EdgeGATConv(in_feats=16, edge_feats=1, out_feats=32, num_heads=num_heads, allow_zero_in_degree=True)
        self.linear1 = nn.Sequential(nn.Linear(32, 16),
                                     nn.LeakyReLU(),
                                     nn.Linear(16, 4),
                                     nn.LeakyReLU(),
                                     nn.Linear(4, 1))
        self.linear2 = nn.Linear(num_heads, 1)


    def forward(self, Graph, h: list, L: list, W, P, N):
        numberOfJobs = len(h)
        numberOfMachines = Graph.shape[1] - numberOfJobs
        jobs = list(zip(h, L))
        jobList = torch.tensor(sorted(jobs, key=lambda x: x[0], reverse=True))
        otherFeatures = torch.tensor([W, P, N, 1])  # 1代表这个feature属于machine

        # jobFeatures形状为 n*6(h_i,L_i,W_i,P_i,n_i,1) 1代表这个feature属于machine
        jobFeatures = torch.cat((jobList, otherFeatures.unsqueeze(1).t().expand(jobList.size(0), -1)), dim=1)

        # machineFeatures填充全为2，大小为numberOfMachines*6
        machineFeatures = torch.ones((numberOfMachines, 6)) * 2

        # terminalFeatures填充全为3，大小为1*6
        terminalFeature = torch.ones((1, 6)) * 3

        node_features = torch.cat((jobFeatures, machineFeatures, terminalFeature), dim=0)


        # 在Graph右侧增加一列0，代表terminal
        add_terminal_tensor = torch.zeros(numberOfJobs, 1)
        Graph_with_terminal = torch.cat((Graph, add_terminal_tensor), dim=1)

        # 下方把machine连terminal的位置填1，其余填0组成方阵
        numberOfJMT = Graph_with_terminal.shape[1]
        tmp_tensor_left = torch.zeros((numberOfJMT - numberOfJobs, numberOfJMT - 1))
        tmp_tensor_right = torch.ones((numberOfJMT - numberOfJobs, 1))
        tmp_tensor_right[-1][-1] = 0
        add_zero_tensor = torch.cat((tmp_tensor_left, tmp_tensor_right), dim=1)
        Graph_edited = torch.cat((Graph_with_terminal, add_zero_tensor), dim=0)

        dgl_Graph = tensor_to_dgl(Graph_edited)


        # machine与machine边的feature为T矩阵对应的时间，其余边的feature都为1
        numberOfEdges = dgl_Graph.num_edges()
        time_matrix = build_time_matrix(jobList, numberOfJobs, W, P, N)[0]
        small_Graph = Graph[:, :numberOfJobs]
        indices = torch.nonzero(small_Graph == 1)
        extracted_elements = time_matrix[indices[:, 0], indices[:, 1]]
        m_to_m_edge_features = extracted_elements.view(-1, 1)
        other_edge_features = torch.ones(numberOfEdges - m_to_m_edge_features.shape[0], 1)
        edge_features = torch.cat((m_to_m_edge_features, other_edge_features), dim=0)


        fst_time_node_feats = self.conv1(dgl_Graph, node_features, edge_features)
        snd_time_node_feats = self.conv2(dgl_Graph, fst_time_node_feats.mean(dim=1), edge_features)
        trd_time_node_feats = self.conv3(dgl_Graph, snd_time_node_feats.mean(dim=1), edge_features)

        feats = self.linear1(trd_time_node_feats).view(numberOfJMT, -1)
        feats = self.linear2(feats)
        feat = torch.max(feats, dim=0).values

        return feat



    def mask(self, Graph: torch.tensor):
        left = torch.ones((Graph.size()[0], Graph.size()[0]))
        right = torch.ones((Graph.size()[0], Graph.size()[1] - Graph.size()[0]))
        row = torch.sum(Graph, dim=1, keepdim=True)
        col = torch.sum(Graph, dim=0, keepdim=True)

        left = left - row - torch.t(row) - col[:, 0:Graph.size()[0]] - torch.t(col[:, 0:Graph.size()[0]])
        left = torch.where(left == 1, torch.tensor(1).float(), torch.tensor(0).float())
        left = left.triu(diagonal=1)
        right -= row

        return torch.cat((left, right), dim=1)



if __name__ == "__main__":
    dd = GraphNN()

    ret = dd.forward(torch.tensor([[0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1]]), [1, 2, 3, 4], [5, 4, 3, 2], 10, 15, 20)

    print(ret)
