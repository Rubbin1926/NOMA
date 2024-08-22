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


BATCH_SIZE = 8  # For testing
reward_multiplicative_factor = 500


def build_time_matrix(jobList, numberOfJobs, W, P, N) -> tuple[torch.Tensor, torch.Tensor]:
    """构建Time矩阵，下半部分全为0"""
    batch_size = jobList.shape[0]
    max_job = jobList.shape[1]
    time_matrix = torch.zeros((batch_size, max_job, max_job))
    time_list = torch.zeros((batch_size, 1, max_job))

    for i in range(batch_size):
        for j in range(numberOfJobs[i].item()):
            R_OMA = W[i] * math.log(1 + jobList[i, j, 0] * P[i] / N[i])
            time_OMA = jobList[i, j, 1] / R_OMA
            time_matrix[i, j, j] = time_OMA
            time_list[i, 0, j] = time_OMA
            for k in range(j + 1, numberOfJobs[i].item()):
                R_NOMA = W[i] * math.log(1 + jobList[i, j, 0] * P[i] / (jobList[i, k, 0] * P[i] + N[i]))
                time_NOMA = jobList[i, k, 1] / R_NOMA
                time_matrix[i, j, k] = max(time_NOMA, time_OMA)

    return time_matrix, time_list


def mask(Graph: torch.Tensor, golden_mask: torch.Tensor) -> torch.Tensor:
    """
    Return the mask of the graph
    输入的Graph是一个batch_size * numberOfJobs * (numberOfJobs + numberOfMachines)的三维张量
    torch.Tensor: 最初始的mask
    输出的mask是一个batch_size * numberOfJobs * (numberOfJobs + numberOfMachines)的三维张量
    """
    numberOfJobs = Graph.shape[-2]
    numberOfMachines = Graph.shape[-1] - numberOfJobs
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
    ret = (torch.cat((left, right), dim=2) * golden_mask.reshape_as(Graph)).bool()
    return ret

def sample_env(batch_size: list) -> tuple:
    batch_size = batch_size[0] if isinstance(batch_size, (list, torch.Size)) else batch_size

    min_job_range, max_job_range = 2, 10
    min_machine_range, max_machine_range = 2, 6

    numberOfJobs = torch.tensor([[random.randint(min_job_range, max_job_range)] for _ in range(batch_size)])
    numberOfMachines = torch.tensor([[random.randint(min_machine_range, max_machine_range)] for _ in range(batch_size)])
    max_job = max_job_range
    max_machine = max_machine_range
    # max_job = torch.max(numberOfJobs).item()
    # max_machine = torch.max(numberOfMachines).item()
    # numberOfJobs = torch.tensor([8]).repeat(batch_size, 1)
    # numberOfMachines = torch.tensor([2]).repeat(batch_size, 1)
    print("sample_env: numberOfJobs:", numberOfJobs.flatten().shape,
          "numberOfMachines:", numberOfMachines.flatten().shape)
    print("max_job:", max_job, "max_machine:", max_machine)

    def h_distribution():
        d = random.uniform(0.1, 0.5)
        tmp0 = (128.1 + 37.6 * math.log(d, 10)) / 10
        tmp1 = 10 ** tmp0
        _h = 1 / tmp1
        return _h, d

    # h与d负相关
    h_and_d = torch.zeros((batch_size, max_job, 2))
    for i in range(batch_size):
        h_and_d[i, :numberOfJobs[i].item(), :] = torch.tensor([sorted([h_distribution() for _ in range(numberOfJobs[i].item())], reverse=True, key=lambda x: x[0])])
    h, norm_h = h_and_d[:, :, 0], h_and_d[:, :, 1]

    L = torch.zeros((batch_size, max_job))
    for i in range(batch_size):
        L[i, :numberOfJobs[i].item()] = torch.tensor([random.randint(1, 1024) for _ in range(numberOfJobs[i].item())])
    norm_L = L.clone() / 200


    W = torch.tensor([[180 / numberOfMachines[i].item() * 1000] for i in range(batch_size)])
    norm_W = W.clone() / 20000

    P = torch.tensor([[0.1] for _ in range(batch_size)])
    norm_P = P.clone()

    N = torch.tensor([[(10 ** (-174 / 10)) / 1000 * (180 / numberOfMachines[i].item() * 1000)] for i in range(batch_size)])
    norm_N = norm_W.clone() / 5
    return h, L, W, P, N, norm_h, norm_L, norm_W, norm_P, norm_N, numberOfJobs, numberOfMachines, max_job, max_machine


def action_to_tensor(Action: torch.Tensor, td: TensorDict) -> torch.Tensor:
    device = Action.device
    max_job = td["max_job"][0].item()
    max_machine = td["max_machine"][0].item()
    valid_indices = Action != -1
    actionTensor = torch.zeros((Action.size(0), max_job*(max_job+max_machine)), device=device)
    actionTensor[torch.arange(Action.size(0), device=device)[valid_indices], Action[valid_indices]] = 1
    return actionTensor.reshape((Action.size(0), max_job, max_job+max_machine))


class NOMAGenerator(Generator):
    def __init__(self):
        super().__init__()
        # print("###Generator###")

    def _generate(self, batch_size, **kwargs) -> TensorDict:
        # print("generate")
        h, L, W, P, N, norm_h, norm_L, norm_W, norm_P, norm_N, numberOfJobs, numberOfMachines, max_job, max_machine = sample_env(batch_size=batch_size)
        bs = batch_size[0] if isinstance(batch_size, (list, torch.Size)) else batch_size
        Graph = torch.zeros((bs, max_job, max_job+max_machine))
        jobList = torch.stack((h, L), dim=-1)
        T, T_list = build_time_matrix(jobList, numberOfJobs, W, P, N)

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
                "numberOfJobs": numberOfJobs,
                "numberOfMachines": numberOfMachines,
                "max_job": torch.tensor([max_job]).repeat(bs, 1),
                "max_machine": torch.tensor([max_machine]).repeat(bs, 1),
            },
            batch_size=batch_size,
        )

    def generate(self, batch_size, **kwargs):
        return self._generate(batch_size, **kwargs)


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
        origin_action = td["action"]
        tensor_action = action_to_tensor(origin_action, td)
        Graph = td["Graph"]
        Graph += tensor_action
        available = mask(Graph, td["golden_mask"])
        done = torch.sum(available.reshape(td.batch_size[0], -1), dim=(-1)) == 0
        td.update(
            {
                "Graph": Graph,
                "action_mask": available.reshape(td.batch_size[0], -1),
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
        print("###Reset###")

        init_Graph = td["Graph"] if td is not None else None
        device = init_Graph.device if init_Graph is not None else self.device
        self.to(device)
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        td = self.generator.generate(batch_size=batch_size)

        action_mask = torch.zeros_like(init_Graph)
        for i in range(batch_size[0]):
            _numberOfJobs = td["numberOfJobs"][i].item()
            _numberOfMachines = td["numberOfMachines"][i].item()
            max_job = td["max_job"][i].item()

            left_part = torch.ones((_numberOfJobs, _numberOfJobs), device=device).triu(diagonal=1)
            right_part = torch.ones((_numberOfJobs, _numberOfMachines), device=device)

            action_mask[i, :_numberOfJobs, :_numberOfJobs] = left_part
            action_mask[i, :_numberOfJobs, max_job:max_job + _numberOfMachines] = right_part

            if _numberOfJobs < max_job:
                action_mask[i, _numberOfJobs:, -1] = 1

        td.update({
                "action_mask": action_mask.reshape(batch_size[0], -1).bool(),
                "golden_mask": action_mask.reshape(batch_size[0], -1).bool(),
            },)

        return td


    def _make_spec(self, generator: NOMAGenerator):
        self.observation_spec = CompositeSpec(
            Graph=UnboundedContinuousTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            h=UnboundedContinuousTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            L=UnboundedDiscreteTensorSpec(
                shape=(1),
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
                shape=(1),
                dtype=torch.float32,
            ),
            norm_L=UnboundedDiscreteTensorSpec(
                shape=(1),
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
        action_mask = mask(Graph, td["golden_mask"]).reshape(td.batch_size[0], -1)
        td.update(
            {
                "action_mask": action_mask.reshape(td.batch_size[0], -1),
            },
        )
        return td

    def calculate_time_nodummy(self, td: TensorDict) -> torch.Tensor:
        Graph, T, T_list = td["Graph"], td["T"], td["T_list"]
        bs, numberOfJobs = T_list.shape[0], T_list.shape[-1]
        Graph = Graph.reshape(bs, numberOfJobs, -1)
        numberOfMachines = Graph.shape[-1] - numberOfJobs
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
        bs, numberOfJobs = T_list.shape[0], T_list.shape[-1]
        Graph = Graph.reshape(bs, numberOfJobs, -1)
        T_list = T_list.reshape(-1, 1, numberOfJobs)

        row = torch.sum(Graph, dim=-1).reshape_as(T_list)
        totalTime_dummy = torch.sum((1 - row) * T_list, dim=-1).flatten()

        ret, _ = torch.max(torch.stack((totalTime_dummy, self.calculate_time_nodummy(td))), dim=0)

        # 注意此结果为正的时间
        return reward_multiplicative_factor * ret

    def step_to_end_from_actions(self, td: TensorDict, actions: torch.Tensor) -> TensorDict:
        zero_Graph = torch.zeros_like(td["Graph"])
        actions = actions.t()
        td.update({"Graph": zero_Graph})

        for i in range(actions.shape[0]):
            td.update({"action": actions[i]})
            td = self._step(td)

        return td

    def next_mask(self, old_mask: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # old_mask.shape: (batch_size, numberOfJobs*(numberOfJobs + numberOfMachines))
        # action.shape: (batch_size, numberOfJobs, numberOfJobs + numberOfMachines)
        # Output: (batch_size, numberOfJobs*(numberOfJobs + numberOfMachines))
        action = action.reshape_as(old_mask)
        return old_mask - action


if __name__ == "__main__":
    env = NOMAenv()


    def my_random_policy(td):
        """Helper function to select a random action from available actions or None if no action is available"""
        action_mask = td["action_mask"].float()
        actions = []
        for mask in action_mask:
            if mask.sum() == 0:
                actions.append(-1)
            else:
                action = torch.multinomial(mask, 1).squeeze(-1)
                actions.append(action.item())
        td.set("action", torch.tensor(actions))
        return td

    reward, td, actions = rollout(env, env.reset(batch_size=[BATCH_SIZE]), my_random_policy)
    print(reward)
    print(td)
    print("actions: ", actions)
    print(env.step_to_end_from_actions(td.clone(), actions)["reward"])

