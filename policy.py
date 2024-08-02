import dgl
import torch
from torch import nn
import torch.nn.functional as F
import math
from dgl.nn import EdgeGATConv
from torch.distributions import Categorical
from env import numberOfJobs, numberOfMachines, BATCH_SIZE

"""
TensorDict(
    fields={
        Graph: Tensor(shape=torch.Size([2, 24]), device=cpu, dtype=torch.float32, is_shared=False),
        L: Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.int64, is_shared=False),
        N: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        P: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        T: Tensor(shape=torch.Size([2, 4, 4]), device=cpu, dtype=torch.float32, is_shared=False),
        T_list: Tensor(shape=torch.Size([2, 1, 4]), device=cpu, dtype=torch.float32, is_shared=False),
        W: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int64, is_shared=False),
        action_mask: Tensor(shape=torch.Size([2, 24]), device=cpu, dtype=torch.bool, is_shared=False),
        done: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.bool, is_shared=False),
        h: Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
        jobList: Tensor(shape=torch.Size([2, 4, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        reward: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([2]),
    device=None,
    is_shared=False)
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        print("###NOMAInitEmbedding###")
        super(NOMAInitEmbedding, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=12, nhead=3, dim_feedforward=512, dropout=0,
                                                   batch_first=True)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.encoder = nn.Sequential(nn.LayerNorm(12),
                                     transformer_encoder,
                                     nn.Linear(12, embed_dim, linear_bias),
                                     nn.LayerNorm(embed_dim),
                                     nn.ReLU()).to(device)

    def forward(self, td):
        # Input: td
        # Output: [batch_size, num_nodes, embed_dim]

        h, L, W, P, N, _tensor = td["h"], td["L"], td["W"], td["P"], td["N"], torch.zeros_like(td["W"])
        concatenated_tensor = torch.cat((h, L, W, P, N, _tensor), dim=1).to(device)
        encoder_output = self.encoder(concatenated_tensor)
        return encoder_output.unsqueeze(1).repeat(1, numberOfJobs*(numberOfJobs+numberOfMachines), 1)


class NOMAContext(nn.Module):
    def __init__(self, embed_dim,  linear_bias=True):
        print("###NOMAContext###")
        super(NOMAContext, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear((numberOfJobs*(numberOfJobs + numberOfMachines))*embed_dim, embed_dim, linear_bias)

    def forward(self, embeddings, td):
        # embeddings: [batch_size, num_nodes, embed_dim]
        # td: td
        # Output: [batch_size, embed_dim]

        embeddings = self.norm(embeddings)
        embeddings = embeddings.view(embeddings.shape[0], -1)
        return self.linear(embeddings)


class NOMADynamicEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        print("###NOMADynamicEmbedding###")
        super(NOMADynamicEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0
