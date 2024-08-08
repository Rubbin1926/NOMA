import dgl
import torch
from torch import nn
import torch.nn.functional as F
import math
from dgl.nn import EdgeGATConv
from env import numberOfJobs, numberOfMachines, BATCH_SIZE
from tensordict.tensordict import TensorDict
from rl4co.models.rl.common.critic import CriticNetwork

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
        norm_L: Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
        norm_N: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        norm_P: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        norm_W: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
        norm_h: Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
        reward: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
        terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([2]),
    device=None,
    is_shared=False)
"""


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor_to_dgl(Graph: torch.Tensor) -> dgl.DGLGraph:  # 输入进来的tensor with batch是处理成方阵的Graph
    """
    用了dgl自带的batch函数

    经测试，与下文函数的结果经过dgl.nn后结果一致，并且此函数for循环导致速度更慢
    """
    graph_list = []
    num_nodes = numberOfJobs+numberOfMachines
    for i in range(Graph.shape[0]):
        DGLgraph = dgl.DGLGraph()
        DGLgraph.add_nodes(num_nodes)
        indices = torch.nonzero(Graph[i]).t().view(2, -1)
        DGLgraph.add_edges(indices[0], indices[1])
        graph_list.append(DGLgraph)

    return dgl.batch(graph_list)


def tensor_to_dgl(Graph: torch.Tensor) -> dgl.DGLGraph:  # 输入进来的tensor with batch是处理成方阵的Graph
    num_nodes = Graph.shape[0] * (numberOfJobs+numberOfMachines)

    # 处理为dgl接受的格式
    indices = torch.nonzero(Graph)
    indices += (indices[:, 0] * (numberOfJobs+numberOfMachines)).reshape(indices.shape[0], 1)
    indices = indices[:, 1:].t().view(2, -1)

    DGLgraph = dgl.graph((indices[0], indices[1]), num_nodes=num_nodes)

    return DGLgraph


class GraphNN(nn.Module):
    def __init__(self, embed_dim):
        super(GraphNN, self).__init__()
        print("my GNN")

        num_heads = 3

        self.conv0 = EdgeGATConv(in_feats=5, edge_feats=1, out_feats=16,
                                 num_heads=num_heads, allow_zero_in_degree=True)
        self.conv1 = EdgeGATConv(in_feats=16, edge_feats=1, out_feats=embed_dim,
                                 num_heads=num_heads, allow_zero_in_degree=True)

        self.node_norm = nn.LayerNorm(5, eps=1e-5)
        self.linear = nn.Linear((numberOfJobs+numberOfMachines)*embed_dim, embed_dim)

    def forward(self, td: TensorDict) -> torch.Tensor:
        # td: td
        # Output: [batch_size, embed_dim]

        device = td["Graph"].device
        self.to(device)

        # 使用手动Norm过的数据
        Graph, h, L, W, P, N = td["Graph"], td["norm_h"], td["norm_L"], td["norm_W"], td["norm_P"], td["norm_N"]
        bs = td.batch_size[0]
        Graph = Graph.reshape(bs, numberOfJobs, numberOfJobs+numberOfMachines)

        square_Graph = torch.zeros(bs, numberOfJobs+numberOfMachines, numberOfJobs+numberOfMachines).to(device)
        square_Graph[:, :numberOfJobs, :numberOfJobs + numberOfMachines] = Graph
        dgl_Graph = tensor_to_dgl(square_Graph)


        otherFeatures = torch.cat((W, P, N), dim=1).unsqueeze(1).repeat(1, numberOfJobs, 1)

        # jobFeatures形状为 B*numberOfJobs*5(h_i,L_i,W_i,P_i,n_i)
        jobList = torch.stack((h, L), dim=-1)
        jobFeatures = torch.cat((jobList, otherFeatures), dim=-1)

        # machineFeatures填充全为0，大小为B*numberOfMachines*5
        machineFeatures = torch.zeros((bs, numberOfMachines, 5)).to(device)

        nodeFeatures = torch.cat((jobFeatures, machineFeatures), dim=1).reshape(-1, 5)
        nodeFeatures = self.node_norm(nodeFeatures)


        # 先将T矩阵填充到G方阵大小，然后根据index直接取出对应的元素作为边的数值
        T_matrix = torch.zeros_like(square_Graph)
        T_matrix[:, :numberOfJobs, :numberOfJobs] = td["T"]

        edge_indices = torch.nonzero(square_Graph)
        batch_idx, row_idx, col_idx = edge_indices[:, 0], edge_indices[:, 1], edge_indices[:, 2]
        edgeFeatures = T_matrix[batch_idx, row_idx, col_idx].reshape(-1, 1)


        zro_time_node_feats = self.conv0(dgl_Graph, nodeFeatures, edgeFeatures)
        zro_time_node_feats = F.leaky_relu(zro_time_node_feats)
        fst_time_node_feats = self.conv1(dgl_Graph, zro_time_node_feats.mean(dim=1), edgeFeatures)
        fst_time_node_feats = F.leaky_relu(fst_time_node_feats)

        conv_out = fst_time_node_feats.mean(dim=1).reshape(bs, -1)
        conv_out = self.linear(conv_out)
        conv_out = F.leaky_relu(conv_out)

        return conv_out


class MyCriticNetwork(CriticNetwork):
    def __init__(self, embed_dim):
        super(CriticNetwork, self).__init__()
        print("my critic network")

        hidden_dim = 256

        self.GNN = GraphNN(embed_dim=embed_dim)
        self.linear = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim, eps=1e-5),
                                    nn.LeakyReLU(),
                                    nn.Linear(hidden_dim, 1))

    def forward(self, td):
        # td: td
        # Output: [batch_size, 1]

        device = td["Graph"].device
        self.to(device)

        gnn_output = self.GNN(td)
        output = self.linear(gnn_output)

        return output


# All For Test
class NOMAInitEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=True):
        print("###NOMAInitEmbedding###")
        super(NOMAInitEmbedding, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=numberOfJobs+numberOfMachines, nhead=2, dim_feedforward=512, dropout=0,
                                                   batch_first=True, layer_norm_eps=1e-5)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.encoder = nn.Sequential(transformer_encoder,
                                     nn.Linear(numberOfJobs+numberOfMachines, embed_dim, linear_bias),
                                     nn.LayerNorm(embed_dim, eps=1e-5),
                                     nn.LeakyReLU())
        self.linear = nn.Linear(7*embed_dim, (numberOfJobs*(numberOfJobs+numberOfMachines))*embed_dim, linear_bias)

    def forward(self, td: TensorDict) -> torch.Tensor:
        # Input: td
        # Output: [batch_size, num_nodes, embed_dim]

        device = td["Graph"].device
        bs = td.batch_size[0]
        self.to(device)

        h, L, W, P, N = td["norm_h"], td["norm_L"], td["norm_W"], td["norm_P"], td["norm_N"]
        feats_tensor = torch.zeros((bs, 7, numberOfJobs+numberOfMachines), device=device)

        feats_tensor[:, 0, numberOfJobs:] = 1
        feats_tensor[:, 1, :numberOfJobs] = 1

        feats_tensor[:, 2, :numberOfJobs] = h
        feats_tensor[:, 3, :numberOfJobs] = L

        feats_tensor[:, 4, numberOfJobs:] = W
        feats_tensor[:, 5, numberOfJobs:] = P
        feats_tensor[:, 6, numberOfJobs:] = N

        encoder_output = self.linear(self.encoder(feats_tensor).reshape(bs, -1))
        return encoder_output.reshape(bs, numberOfJobs*(numberOfJobs+numberOfMachines), -1)


class NOMAContext(nn.Module):
    def __init__(self, embed_dim,  linear_bias=True):
        print("###NOMAContext###")
        super(NOMAContext, self).__init__()

        self.for_embedding = nn.Sequential(nn.Linear(embed_dim, embed_dim, linear_bias),
                                           nn.LayerNorm(embed_dim, eps=1e-5),
                                           nn.LeakyReLU())

        self.GNN = GraphNN(embed_dim=embed_dim)

        self.linear = nn.Linear(embed_dim, embed_dim, linear_bias)

    def forward(self, embeddings: torch.Tensor, td: TensorDict) -> torch.Tensor:
        # embeddings: [batch_size, num_nodes, embed_dim]
        # td: td
        # Output: [batch_size, embed_dim]

        device = td["Graph"].device
        self.to(device)

        embeddings = embeddings.mean(dim=1)
        embeddings = self.for_embedding(embeddings)
        # for name, param in self.GNN.named_parameters():
        #     print(f"Parameter: {param.device}")
        # print(td["W"].device)
        gnn_output = self.GNN(td)

        output = F.leaky_relu(self.linear(embeddings + gnn_output))

        return output


class NOMADynamicEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        print("###NOMADynamicEmbedding###")
        super(NOMADynamicEmbedding, self).__init__()

    def forward(self, td: TensorDict):
        return 0, 0, 0
