import time
from tensordict.tensordict import TensorDict
import dgl
import torch
from torch import nn
import torch.nn.functional as F
import math
from dgl.nn import EdgeGATConv
from torch.distributions import Categorical
from env import numberOfJobs, numberOfMachines, BATCH_SIZE, NOMAenv, mask
from rl4co.utils.decoding import rollout, random_policy
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 64

env = NOMAenv()
env.reset(batch_size=[BATCH_SIZE])

reward, td, actions = rollout(env, env.reset(batch_size=[BATCH_SIZE]), random_policy)

def generate_mask_list(Mask):
    mask_indices = torch.nonzero(Mask)  # 获取mask中为1的元素的索引位置
    tensors = []

    for index in mask_indices:
        tensor = torch.zeros_like(Mask)  # 创建和mask大小相同的全0矩阵
        tensor[index[0], index[1]] = 1  # 将对应位置的元素设置为1
        tensors.append(tensor)

    return tensors


def find(td, Dataset, env):
    device = td['Graph'].device
    Graph = torch.zeros_like(td['Graph']).reshape(numberOfJobs, numberOfJobs+numberOfMachines)
    Mask = mask(Graph).reshape_as(Graph)

    if torch.sum(Mask) == 0:
        td_tmp = {"Graph": Graph, "T": td["T"], "T_list": td["T_list"]}
        Dataset[Graph] = (torch.zeros_like(Graph),
                          Graph,
                          env.calculate_time_dummy(td_tmp))
        return Graph, env.calculate_time_dummy(td_tmp)

    if Graph in Dataset:
        return Dataset[Graph][1], Dataset[Graph][2]

    V_star = -1e16
    A_star = None
    G_star = None
    mask_list = generate_mask_list(Mask)

    for action in mask_list:
        td_tmp = {"Graph": Graph+action, "T": td["T"], "T_list": td["T_list"]}
        G_, V_ = find(td_tmp, Dataset, env)
        if V_ > V_star:
            V_star = V_
            A_star = action
            G_star = G_

    Dataset[Graph] = (A_star,
                      G_star,
                      V_star)
    return G_star, V_star

first_batch_index = 0
first_batch_tensor_dict = TensorDict({key: value[first_batch_index] for key, value in td.items()})

dat = {}
find(first_batch_tensor_dict, dat, env)
print(dat)
