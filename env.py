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


numberOfJobs = 8
numberOfMachines = 2
BATCH_SIZE = 2
assert (numberOfJobs+numberOfMachines) % 2 == 0, "(numberOfJobs+numberOfMachines)需要是偶数！"


def build_time_matrix(jobList, W, P, N) -> tuple[torch.Tensor, torch.Tensor]:
    """构建Time矩阵，下半部分全为0"""
    batch_size = jobList.shape[0]
    time_matrix = torch.zeros((batch_size, numberOfJobs, numberOfJobs))
    time_list = torch.zeros((batch_size, 1, numberOfJobs))

    for i in range(batch_size):
        for j in range(numberOfJobs):
            R_OMA = W[i] * math.log(1 + jobList[i, j, 0] * P[i] / N[i])
            time_OMA = jobList[i, j, 1] / R_OMA
            time_matrix[i, j, j] = time_OMA
            time_list[i, 0, j] = time_OMA
            for k in range(j + 1, numberOfJobs):
                R_NOMA = W[i] * math.log(1 + jobList[i, j, 0] * P[i] / (jobList[i, k, 0] * P[i] + N[i]))
                time_NOMA = jobList[i, k, 1] / R_NOMA
                time_matrix[i, j, k] = max(time_NOMA, time_OMA)

    return time_matrix, time_list


def mask(Graph: torch.Tensor) -> torch.Tensor:
    """
    Return the mask of the graph
    输入的Graph是一个batch_size * (numberOfJobs * (numberOfJobs + numberOfMachines))的二维张量
    输出的mask是一个batch_size * (numberOfJobs * (numberOfJobs + numberOfMachines))的二维张量
    """
    Graph = Graph.reshape(-1, numberOfJobs, numberOfJobs + numberOfMachines)
    batch_size = Graph.shape[0]
    graph_shape = Graph.shape[1:]

    left = torch.ones((batch_size, graph_shape[0], graph_shape[0])).to(Graph.device)
    right = torch.ones((batch_size, graph_shape[0], graph_shape[1] - graph_shape[0])).to(Graph.device)

    row = torch.sum(Graph, dim=2, keepdim=True)
    col = torch.sum(Graph, dim=1, keepdim=True)

    left = left - row - torch.transpose(row, 1, 2) - col[:, :, 0:graph_shape[0]] - torch.transpose(col[:, :, 0:graph_shape[0]], 1, 2)
    left = torch.where(left == 1, torch.tensor(1), torch.tensor(0))
    left = left.triu(diagonal=1)
    right -= row

    ret = torch.cat((left, right), dim=2).bool().reshape(batch_size, -1)

    return ret


def sample_env(batch_size: list) -> tuple:
    batch_size = batch_size[0] if isinstance(batch_size, list) else batch_size
    def h_distribution():
        d = random.uniform(0.1, 0.5)
        tmp0 = (128.1 + 37.6 * math.log(d, 10)) / 10
        tmp1 = 10 ** tmp0
        _h = 1 / tmp1
        return _h, d

    # h与d负相关
    h_and_d = torch.tensor([sorted([h_distribution() for _ in range(numberOfJobs)], reverse=True, key=lambda x: x[0]) for _ in range(batch_size)])
    h, norm_h = h_and_d[:, :, 0], h_and_d[:, :, 1]

    L = torch.tensor(np.random.randint(1, 1024, size=(batch_size, numberOfJobs)).tolist())
    norm_L = L / 200

    W = torch.tensor([[180 / numberOfMachines * 1000] for _ in range(batch_size)])
    norm_W = W / 20000

    P = torch.tensor([[0.1] for _ in range(batch_size)])
    norm_P = P.clone()

    N = torch.tensor([[(10 ** (-174 / 10)) / 1000 * (180 / numberOfMachines * 1000)] for _ in range(batch_size)])
    norm_N = norm_W.clone() / 5
    return h, L, W, P, N, norm_h, norm_L, norm_W, norm_P, norm_N


def action_to_tensor(Action: torch.Tensor) -> torch.Tensor:
    # rowPosition = Action // (numberOfJobs + numberOfMachines)
    # colPosition = Action % (numberOfJobs + numberOfMachines)
    # actionTensor = torch.zeros((Action.size(0), numberOfJobs, numberOfJobs + numberOfMachines))
    # actionTensor[torch.arange(Action.size(0)), rowPosition, colPosition] = 1
    actionTensor = torch.zeros((Action.size(0), numberOfJobs*(numberOfJobs + numberOfMachines))).to(Action.device)
    actionTensor[torch.arange(Action.size(0)), Action] = 1
    return actionTensor


class NOMAGenerator(Generator):
    def __init__(self):
        pass
        # print("###Generator###")

    def _generate(self, batch_size) -> TensorDict:
        h, L, W, P, N, norm_h, norm_L, norm_W, norm_P, norm_N = sample_env(batch_size=batch_size)
        bs = batch_size[0] if isinstance(batch_size, list) else batch_size
        Graph = torch.zeros((bs, numberOfJobs*(numberOfJobs + numberOfMachines)))
        jobList = torch.stack((h, L), dim=-1)
        T, T_list = build_time_matrix(jobList, W, P, N)

        return TensorDict(
            {
                "Graph": Graph,
                "h": h,
                "L": L,
                "W": W,
                "P": P,
                "N": N,
                "jobList": jobList,
                "T": T,
                "T_list": T_list,
                "norm_h": norm_h,
                "norm_L": norm_L,
                "norm_W": norm_W,
                "norm_P": norm_P,
                "norm_N": norm_N,
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
        # print("__________###Step###__________")

        action = td["action"]
        Graph = td["Graph"]

        Graph += (mask(Graph)) * action_to_tensor(action)
        available = mask(Graph)
        done = torch.sum(available, dim=(-1)) == 0

        td.update(
            {
                "Graph": Graph,
                "action_mask": available,
                "done": done,
            },
        )

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        # reward = torch.zeros_like(done) * done
        reward = -self.calculate_time_dummy(td)
        td.update(
            {
                "reward": reward,
            },
        )

        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # print("###Reset###")

        init_Graph = td["Graph"] if td is not None else None
        # if batch_size is None:
        #     batch_size = self.batch_size if init_Graph is None else BATCH_SIZE
        device = init_Graph.device if init_Graph is not None else self.device
        self.to(device)
        # if init_Graph is None:
        #     bs = batch_size[0] if isinstance(batch_size, list) else batch_size
        #     init_Graph = torch.zeros((bs, numberOfJobs*(numberOfJobs + numberOfMachines)))
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        return TensorDict(
            {
                "action_mask": mask(init_Graph),
            },
            batch_size=batch_size,
        )


    def _make_spec(self, generator: NOMAGenerator):
        self.observation_spec = CompositeSpec(
            Graph=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(numberOfJobs*(numberOfJobs + numberOfMachines)),
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
            norm_h=UnboundedContinuousTensorSpec(
                shape=(numberOfJobs),
                dtype=torch.float32,
            ),
            norm_L=UnboundedDiscreteTensorSpec(
                shape=(numberOfJobs),
                dtype=torch.float32,
            ),
            norm_W=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            norm_P=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            norm_N=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            low=0,
            high=1,
            shape=(1,),
            dtype=torch.int64,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1), dtype=torch.bool)

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        # print("###get_reward###")
        # 在一段trajectory结束后，才会调用
        # print("reward", -self.calculate_time_dummy(td))

        return -self.calculate_time_dummy(td)

    def get_action_mask(self, td: TensorDict) -> TensorDict:
        Graph = td["Graph"]
        action_mask = mask(Graph)
        td.update(
            {
                "action_mask": action_mask,
            },
        )
        return td

    def calculate_time_nodummy(self, td: TensorDict) -> torch.Tensor:
        Graph, T, T_list = td["Graph"], td["T"], td["T_list"]
        Graph = Graph.reshape(-1, numberOfJobs, numberOfJobs + numberOfMachines)
        T = T.reshape(-1, numberOfJobs, numberOfJobs)
        T_list = T_list.reshape(-1, 1, numberOfJobs)

        G_tmp = Graph[:, :, 0:numberOfJobs]
        row = torch.sum(G_tmp, dim=-1)
        col = torch.sum(G_tmp, dim=-2)

        totalTime_OMA_fake = (1 - row - col).unsqueeze(dim=-2) * T_list
        totalTime_NOMA_fake = torch.sum(G_tmp * T, dim=-2).unsqueeze(dim=-2)
        totalTime_fake = totalTime_OMA_fake + totalTime_NOMA_fake

        ret, _ = torch.max(totalTime_fake @ (Graph[:, :, numberOfJobs: (numberOfJobs + numberOfMachines)]), dim=-1)

        return ret.flatten()

    def calculate_time_dummy(self, td: TensorDict) -> torch.Tensor:
        Graph, T_list = td["Graph"], td["T_list"]
        Graph = Graph.reshape(-1, numberOfJobs, numberOfJobs + numberOfMachines)
        T_list = T_list.reshape(-1, 1, numberOfJobs)

        row = torch.sum(Graph, dim=-1).reshape_as(T_list)
        totalTime_dummy = torch.sum((1 - row) * T_list, dim=-1).flatten()

        ret, _ = torch.max(torch.stack((totalTime_dummy, self.calculate_time_nodummy(td))), dim=0)
        return ret


if __name__ == "__main__":
    env = NOMAenv()
    env.reset(batch_size=[BATCH_SIZE])
    reward, td, actions = rollout(env, env.reset(batch_size=[BATCH_SIZE]), random_policy)
    print(reward)
    print(td)
    print(actions)
