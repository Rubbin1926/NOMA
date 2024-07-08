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
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import Generator, get_sampler

numberOfJobs = 4
numberOfMachines = 2
BATCH_SIZE = 1


def mask(Graph: torch.Tensor) -> torch.Tensor:
    batch_size = Graph.shape[0]
    graph_shape = Graph.shape[1:]

    left = torch.ones((batch_size, graph_shape[0], graph_shape[0]))
    right = torch.ones((batch_size, graph_shape[0], graph_shape[1] - graph_shape[0]))

    row = torch.sum(Graph, dim=2, keepdim=True)
    col = torch.sum(Graph, dim=1, keepdim=True)

    left = left - row - torch.transpose(row, 1, 2) - col[:, :, 0:graph_shape[0]] - torch.transpose(col[:, :, 0:graph_shape[0]], 1, 2)
    left = torch.where(left == 1, torch.tensor(1).float(), torch.tensor(0).float())
    left = left.triu(diagonal=1)
    right -= row

    return torch.cat((left, right), dim=2).bool()


def sample_env(batch_size: int) -> tuple:
    def h_distribution():
        d = random.uniform(0.1, 0.5)
        tmp0 = (128.1 + 37.6 * math.log(d, 10)) / 10
        tmp1 = 10 ** tmp0
        _h = 1 / tmp1
        return _h

    h = torch.tensor([sorted([h_distribution() for _ in range(numberOfJobs)], reverse=True) for _ in range(batch_size)])
    L = torch.tensor(np.random.randint(1, 1024, size=(batch_size, numberOfJobs)).tolist())
    W = torch.tensor([[180 / numberOfMachines * 1000] for _ in range(batch_size)])
    P = torch.tensor([[0.1] for _ in range(batch_size)])
    N = torch.tensor([[(10 ** (-174 / 10)) / 1000 * (180 / numberOfMachines * 1000)] for _ in range(batch_size)])
    return (h, L, W, P, N)


def action_to_tensor(numpyAction):
    if isinstance(numpyAction, torch.Tensor):
        return numpyAction
    rowPosition = numpyAction // (numberOfJobs+numberOfMachines)
    colPosition = numpyAction % (numberOfJobs+numberOfMachines)
    actionTensor = torch.zeros((numberOfJobs, numberOfJobs+numberOfMachines))
    actionTensor[rowPosition][colPosition] = 1
    return actionTensor


class NOMAGenerator(Generator):
    def __init__(self):
        pass

    def _generate(self, batch_size) -> TensorDict:
        h, L, W, P, N = sample_env(batch_size=batch_size)
        Graph = torch.zeros((batch_size, numberOfJobs, numberOfJobs + numberOfMachines))
        return TensorDict(
            {
                "Graph": Graph,
                "h": h,
                "L": L,
                "W": W,
                "P": P,
                "N": N,
            },
            batch_size=batch_size,
        )



class NOMAenv(RL4COEnvBase):
    """NOMA environment"""

    name = "NOMA"

    def __init__(
        self,
        generator: NOMAGenerator = None,
        generator_params: dict = {},
        check_solution=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = NOMAGenerator(**generator_params)
        self.generator = generator
        self.check_solution = check_solution
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        print("###Step###")

        action = td["action"]
        Graph = td["Graph"]

        Graph += mask(Graph) * action_to_tensor(action)
        available = mask(Graph)

        done = torch.sum(available, dim=(-2, -1)) == 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done) * done

        td.update(
            {
                "Graph": Graph,
                "action_mask": available,
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        print("###Reset###")

        init_Graph = td["Graph"] if td is not None else None
        if batch_size is None:
            batch_size = self.batch_size if init_Graph is None else BATCH_SIZE
        device = init_Graph.device if init_Graph is not None else self.device
        self.to(device)
        if init_Graph is None:
            init_Graph = self.generate_data(batch_size=batch_size).to(device)["Graph"]
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Other variables
        h = self.generate_data(batch_size=batch_size).to(device)["h"]
        L = self.generate_data(batch_size=batch_size).to(device)["L"]
        W = self.generate_data(batch_size=batch_size).to(device)["W"]
        P = self.generate_data(batch_size=batch_size).to(device)["P"]
        N = self.generate_data(batch_size=batch_size).to(device)["N"]

        return TensorDict(
            {
                "Graph": init_Graph,
                "h": h,
                "L": L,
                "W": W,
                "P": P,
                "N": N,
                "action_mask": mask(init_Graph),
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: NOMAGenerator):
        self.observation_spec = CompositeSpec(
            Graph=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(numberOfJobs, numberOfJobs + numberOfMachines),
                dtype=torch.int64,
            ),
            h=UnboundedContinuousTensorSpec(
                shape=(numberOfJobs),
                dtype=torch.float32,
            ),
            L=UnboundedDiscreteTensorSpec(
                shape=(numberOfJobs),
                dtype=torch.float32,
            ),
            W=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            P=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            N=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            low=0,
            high=1,
            shape=(numberOfJobs, numberOfJobs + numberOfMachines),
            dtype=torch.int64,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1), dtype=torch.bool)


    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        pass

        # TodoList:
        # 先把TimeMatrix函数改写为batch版本
        # 再把calculate_time_nodummy和calculate_time_dummy改写为batch版本
