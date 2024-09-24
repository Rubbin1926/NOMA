import dgl
import torch

from torch import nn, Tensor
import torch.nn.functional as F
import math
from dgl.nn import EdgeGATConv
from tensordict.tensordict import TensorDict
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import calculate_entropy
from rl4co.utils.pylogger import get_pylogger
import wandb
from typing import *

log = get_pylogger(__name__)

torch.autograd.set_detect_anomaly(True)

def tensor_to_dgl(Graph: Tensor, max_job: int, max_machine: int) -> dgl.DGLGraph:
    # 输入进来的tensor with batch是处理成方阵的Graph

    num_nodes = Graph.shape[0] * (max_job+max_machine)

    # 处理为dgl接受的格式
    indices = torch.nonzero(Graph)
    indices += (indices[:, 0] * (max_job+max_machine)).reshape(indices.shape[0], 1)
    indices = indices[:, 1:].t().view(2, -1)

    DGLgraph = dgl.graph((indices[0], indices[1]), num_nodes=num_nodes)

    return DGLgraph


class myEncoder(nn.Module):
    def __init__(self, embed_dim, linear_bias=True):
        print("###NOMAInitEmbedding###")
        super(myEncoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=10, nhead=2, dim_feedforward=4*embed_dim,
                                                   batch_first=True, layer_norm_eps=1e-5, dropout=0)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.encoder = nn.Sequential(transformer_encoder,
                                     nn.Linear(10, embed_dim, linear_bias),
                                     nn.LayerNorm(embed_dim, eps=1e-5),
                                     nn.LeakyReLU())

        self.linear = nn.Sequential(nn.Linear(embed_dim, embed_dim, linear_bias),
                                    nn.LayerNorm(embed_dim, eps=1e-5),
                                    nn.LeakyReLU())

    def forward(self, td: TensorDict) -> Tensor:
        # td: TensorDict
        # Output: [batch_size, max_job+max_machine, embed_dim]

        device = td["Graph"].device
        bs = td.batch_size[0]

        self.to(device)

        h, L, W, P, N = td["norm_h"], td["norm_L"], td["norm_W"], td["norm_P"], td["norm_N"]
        max_job = h.shape[-1]
        max_machine = td["Graph"].shape[-1] - max_job
        numberOfJobs, numberOfMachines = td["numberOfJobs"], td["numberOfMachines"]

        feats_tensor = torch.zeros((bs, 10, max_job+max_machine), device=device)

        feats_tensor[:, 0, max_job:] = 1
        feats_tensor[:, 1, :max_job] = 1

        feats_tensor[:, 2, :max_job] = h
        feats_tensor[:, 3, :max_job] = L

        feats_tensor[:, 4, max_job:] = W
        feats_tensor[:, 5, max_job:] = P
        feats_tensor[:, 6, max_job:] = N

        jobID = torch.zeros((bs, 1, max_job), device=device)
        indices = torch.arange(1, max_job + 1, device=device).unsqueeze(0).repeat(bs, 1)
        _mask = torch.arange(max_job, device=device).unsqueeze(0) < numberOfJobs
        jobID[:, 0, :] = indices * _mask
        feats_tensor[:, 7, :max_job] = jobID.reshape(bs, max_job)

        machineID = torch.zeros((bs, 1, max_machine), device=device)
        indices = torch.arange(1, max_machine + 1, device=device).unsqueeze(0).repeat(bs, 1)
        _mask = torch.arange(max_machine, device=device).unsqueeze(0) < numberOfMachines
        machineID[:, 0, :] = indices * _mask
        feats_tensor[:, 8, max_job:] = machineID.reshape(bs, max_machine)

        feats_tensor = feats_tensor.permute(0, 2, 1)
        # feats_tensor.shape: [batch_size, max_job+max_machine, 10]

        encoder_output = self.encoder(feats_tensor)

        return self.linear(encoder_output)


class GraphNN(nn.Module):
    def __init__(self, embed_dim):
        super(GraphNN, self).__init__()
        print("my GNN")

        num_heads = 5

        self.conv0 = EdgeGATConv(in_feats=embed_dim, edge_feats=1, out_feats=embed_dim,
                                 num_heads=num_heads, allow_zero_in_degree=True)
        self.conv1 = EdgeGATConv(in_feats=embed_dim, edge_feats=1, out_feats=embed_dim,
                                 num_heads=num_heads, allow_zero_in_degree=True)
        self.conv2 = EdgeGATConv(in_feats=embed_dim, edge_feats=1, out_feats=embed_dim,
                                 num_heads=num_heads, allow_zero_in_degree=True)

        self.linear = nn.Linear(num_heads * embed_dim, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=4*embed_dim,
                                                   batch_first=True, layer_norm_eps=1e-5, dropout=0)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

    def forward(self, td: TensorDict, encoder_output: Tensor) -> Tensor:
        # td: TensorDict
        # encoder_output: [batch_size, max_job+max_machine, embed_dim]
        # Output: [batch_size, max_job+max_machine, embed_dim]

        device = td["Graph"].device
        self.to(device)

        Graph, embed_dim = td["Graph"], encoder_output.shape[-1]
        bs = td.batch_size[0]
        max_job = Graph.shape[-2]
        max_machine = Graph.shape[-1] - max_job
        Graph = Graph.reshape(bs, max_job, max_job+max_machine)

        square_Graph = torch.zeros((bs, max_job+max_machine, max_job+max_machine), device=device)
        square_Graph[:, :max_job, :max_job + max_machine] = Graph
        dgl_Graph = tensor_to_dgl(square_Graph, max_job, max_machine)

        nodeFeatures = encoder_output.reshape(bs*(max_job+max_machine), embed_dim)

        # 先将T矩阵填充到G方阵大小，然后根据index直接取出对应的元素作为边的数值
        T_matrix = torch.zeros_like(square_Graph)
        T_matrix[:, :max_job, :max_job] = td["T"]

        edge_indices = torch.nonzero(square_Graph)
        batch_idx, row_idx, col_idx = edge_indices[:, 0], edge_indices[:, 1], edge_indices[:, 2]
        edgeFeatures = T_matrix[batch_idx, row_idx, col_idx].reshape(-1, 1)

        zro_time_node_feats = self.conv0(dgl_Graph, nodeFeatures, edgeFeatures)
        zro_time_node_feats = F.leaky_relu(zro_time_node_feats)

        fst_time_node_feats = self.conv1(dgl_Graph, zro_time_node_feats.mean(dim=1), edgeFeatures)
        fst_time_node_feats = F.leaky_relu(fst_time_node_feats)

        snd_time_node_feats = self.conv2(dgl_Graph, fst_time_node_feats.mean(dim=1), edgeFeatures)
        snd_time_node_feats = F.leaky_relu(snd_time_node_feats)
        # shape = [batch_size*(max_job+max_machine), num_heads, embed_dim]

        output = self.linear(snd_time_node_feats.reshape(bs*(max_job+max_machine), -1))
        output = output.reshape(bs, max_job+max_machine, embed_dim)

        output = self.transformer_decoder(tgt=output, memory=encoder_output)

        return output


class MyCriticNetwork(CriticNetwork):
    def __init__(self, embed_dim):
        super(CriticNetwork, self).__init__()
        print("my critic network")

        hidden_dim = embed_dim // 2

        self.encoder = myEncoder(embed_dim)
        self.GNN = GraphNN(embed_dim=embed_dim)
        self.linear = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(hidden_dim, 1))

    def forward(self, td: TensorDict) -> torch.Tensor:
        # td: td
        # Output: [batch_size, 1]

        device = td["Graph"].device
        bs = td.batch_size[0]
        self.to(device)

        encoder_output = self.encoder(td)

        gnn_output = self.GNN(td, encoder_output)
        output = self.linear(gnn_output).reshape(bs, -1)
        output = torch.mean(output, dim=-1)

        return output.reshape(bs, -1)


class NOMANet(nn.Module):
    def __init__(self, embed_dim):
        super(NOMANet, self).__init__()
        print("my Policy")

        self.encoder = myEncoder(embed_dim)

        self.GNN0 = GraphNN(embed_dim)
        self.GNN1 = GraphNN(embed_dim)
        self.GNN2 = GraphNN(embed_dim)

        self.linear = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, td: TensorDict) -> Tuple[Tensor, Tensor]:
        device = td["Graph"].device
        bs, max_job = td["Graph"].shape[0], td["Graph"].shape[-2]

        encoder_output = self.encoder(td)

        decoder_output = self.GNN0(td, encoder_output)
        decoder_output = self.GNN1(td, decoder_output)
        decoder_output = self.GNN2(td, decoder_output)

        input1 = self.linear(decoder_output)
        input2 = decoder_output[:, :max_job]
        score = torch.einsum("bmd,bnd->bmn", input1, input2).reshape(bs, -1)

        logits = torch.log(self.softmax(score))

        return logits, td["action_mask"]


class GNNPolicy(nn.Module):
    def __init__(
        self,
        NOMANet: nn.Module,
        temperature: float = 1.0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(GNNPolicy, self).__init__()

        if len(unused_kw) > 0:
            log.error(f"Found {len(unused_kw)} unused kwargs: {unused_kw}")

        self.NOMANet = NOMANet

        # Decoding strategies
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            calc_reward: Whether to calculate the reward
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_hidden: Whether to return the hidden state
            return_init_embeds: Whether to return the initial embeddings
            return_sum_log_likelihood: Whether to return the sum of the log likelihood
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            max_steps: Maximum number of decoding steps for sanity check to avoid infinite loops if envs are buggy (i.e. do not reach `done`)
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logits, mask = self.NOMANet(td)

            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            # 此时td内的action被更新过了
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)
        # logprobs: 所有的actions对应的logprobs，shape = [batch_size, 走多少步]
        # actions: 所有的actions，shape = [batch_size, 走多少步]
        # 剩下两个没动

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))
        # 你终于算reward了，太感动了

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }

        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        # if return_hidden:
        #     outdict["hidden"] = hidden
        # if return_init_embeds:
        #     outdict["init_embeds"] = init_embeds

        return outdict
