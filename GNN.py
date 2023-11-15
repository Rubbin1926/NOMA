import dgl
import torch
from torch import nn
from torch.nn import functional as F
from dgl.nn import TAGConv


def tensor_to_dgl(tensor):
    graph = dgl.DGLGraph()
    num_nodes = tensor.shape[0]
    graph.add_nodes(num_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if tensor[i, j] == 1:
                graph.add_edges(i, j)

    return graph


class GraphNN(nn.Module):
    def __init__(self, Graph: torch.tensor, output_features_number: int):
        super(GraphNN, self).__init__()
        self.Graph = Graph
        self.output_features_number = output_features_number
        self.linear = nn.Linear(output_features_number, Graph.size()[1])
        print(self.Graph)

    def forward(self, h: list, L: list, W, P, N):
        #n
        numberOfJobs = len(h)
        jobs = list(zip(h, L))
        jobList = torch.tensor(sorted(jobs, key=lambda x: x[0], reverse=True))
        otherFeatures = torch.tensor([W, P, N, 1])
        # n x 6 (h_i,L_i,W_i,P_i,n_i, 1 )  1 indicates it is device
        jobFeatures = torch.cat((jobList, otherFeatures.unsqueeze(1).t().expand(jobList.size(0), -1)), dim=1)

        # 将jobFeatures下方部分补全全0
        _, numberOfALL = self.Graph.size()
        num_zeros = numberOfALL - numberOfJobs
        zeros_tensor = torch.zeros((num_zeros, 6))
        #n+d x 6
        jobFeatures = torch.cat((jobFeatures, zeros_tensor), dim=0)
        print(f"""jobFeatures = {jobFeatures}""")

        #将tensor版本的Graph转换成dgl版本
        zeros_tensor = torch.zeros((self.Graph.size()[1] - numberOfJobs, self.Graph.size()[1]))
        Graph = torch.cat((self.Graph.detach(), zeros_tensor), dim=0)

        dgl_Graph = tensor_to_dgl(Graph)

        #调包
        #n+d x f
        conv = TAGConv(6, self.output_features_number)
        Graph_Telda = conv(dgl_Graph, jobFeatures)

        #score = Bilinear (Graph_Telda[:n],Graph_Telda) n x n+d

        Value = self.linear(Graph_Telda[:numberOfJobs])
        Possibility_tmp = Value - (1 - self.mask(self.Graph)) * 1e4
        Possibility = F.softmax(Possibility_tmp.view(-1), dim=0)
        Possibility = Possibility.view(Possibility_tmp.size())

        return Value, Possibility


    def mask(self, Graph: torch.tensor):
        left = torch.ones((Graph.size()[0], Graph.size()[0]))
        right = torch.ones((Graph.size()[0], Graph.size()[1]-Graph.size()[0]))
        row = torch.sum(Graph, dim=1, keepdim=True)
        col = torch.sum(Graph, dim=0, keepdim=True)

        left = left - row - torch.t(row) - col[:, 0:Graph.size()[0]] - torch.t(col[:, 0:Graph.size()[0]])
        left = torch.where(left == 1, torch.tensor(1).float(), torch.tensor(0).float())
        left = left.triu(diagonal=1)
        right -= row

        return torch.cat((left, right), dim=1)



dd = GraphNN(torch.tensor([[0,1,0,0,0,0],
                            [0,0,0,0,1,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0]]), 4)

ret = dd.forward([1,2,3,4],[5,4,3,2],10,15,20)
print(ret)



