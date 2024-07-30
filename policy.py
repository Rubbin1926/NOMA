import dgl
import torch
from torch import nn
import torch.nn.functional as F
import math
from dgl.nn import EdgeGATConv
from torch.distributions import Categorical
from env import numberOfJobs, numberOfMachines, BATCH_SIZE


def tensor_to_dgl(Graph: torch.Tensor) -> dgl.DGLGraph:  # 输入进来的tensor with batch是处理成方阵的Graph
    DGLgraph = dgl.DGLGraph()
    num_nodes = Graph.shape[0] * (numberOfJobs+numberOfMachines)
    DGLgraph.add_nodes(num_nodes)

    # 处理为dgl接受的格式
    indices = torch.nonzero(Graph)
    indices += (indices[:, 0] * (numberOfJobs+numberOfMachines)).reshape(indices.shape[0], 1)
    indices = indices[:, 1:].t().contiguous().view(2, -1)

    DGLgraph.add_edges(indices[0], indices[1])

    return DGLgraph


# All For Test
class NOMAInitEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=True):
        # print("###NOMAInitEmbedding###")

        super(NOMAInitEmbedding, self).__init__()
        node_dim = 1
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        # Input: td
        # Output: [batch_size, num_nodes, embed_dim]
        bs = td["Graph"].shape[0]
        ret = self.init_embed(torch.ones(bs, (numberOfJobs*(numberOfJobs + numberOfMachines)), 1).to(td.device))
        return ret


class NOMAContext(nn.Module):
    def __init__(self, embed_dim,  linear_bias=True):
        # print("###NOMAContext###")

        super(NOMAContext, self).__init__()
        self.linear = nn.Linear((numberOfJobs*(numberOfJobs + numberOfMachines))*embed_dim, embed_dim, linear_bias)

    def forward(self, embeddings, td):
        # embeddings: [batch_size, num_nodes, embed_dim]
        # td: td
        # Output: [batch_size, embed_dim]

        embeddings = embeddings.view(embeddings.shape[0], -1)
        return self.linear(embeddings)


class NOMADynamicEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        # print("###NOMADynamicEmbedding###")

        super(NOMADynamicEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0
