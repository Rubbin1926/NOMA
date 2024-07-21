from typing import Optional
import torch
import torch.nn as nn
import numpy as np
import random
import math

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from rl4co.utils.decoding import rollout, random_policy
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import Generator, get_sampler

from env import numberOfJobs, numberOfMachines, BATCH_SIZE


class NOMAInitEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=True):
        super(NOMAInitEmbedding, self).__init__()
        node_dim = numberOfJobs
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        ret = self.init_embed(td["h"])  # For Test
        return ret


class NOMAContext(nn.Module):
    def __init__(self, embed_dim,  linear_bias=True):
        super(NOMAContext, self).__init__()
        pass

    def forward(self, embeddings, td):
        # embeddings: [batch_size, num_nodes, embed_dim]
        # Output: [batch_size, embed_dim]
        pass


class NOMADynamicEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NOMADynamicEmbedding, self).__init__()
        pass

    def forward(self, td):
        pass
