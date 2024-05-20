import torch
import math
from typing import Optional
import random
import numpy as np
import sys
import gymnasium as gym
from gymnasium import spaces


numberOfJobs = 4
numberOfMachines = 2


def build_time_matrix(jobList, numberOfJobs, W, P, n):
    """构建Time矩阵，下半部分全为0"""
    time_matrix = torch.zeros((numberOfJobs, numberOfJobs))
    time_list = torch.zeros((1, numberOfJobs))

    for j in range(numberOfJobs):
        R_OMA = W * math.log(1 + jobList[j][0] * P / n)
        time_OMA = jobList[j][1] / R_OMA
        time_matrix[j, j] = time_OMA
        time_list[0, j] = time_OMA
        for k in range(j + 1, numberOfJobs):
            R_NOMA = W * math.log(1 + jobList[j][0] * P / (jobList[k][0] * P + n))
            time_NOMA = jobList[k][1] / R_NOMA
            time_matrix[j, k] = max(time_NOMA, time_OMA)

    return time_matrix, time_list


def sample_env():
    # Parameters in paper
    def h_distribution():
        d = random.uniform(0.1, 0.5)
        tmp0 = (128.1 + 37.6 * math.log(d, 10)) / 10
        tmp1 = 10 ** tmp0
        _h = 1 / tmp1
        return _h

    h = torch.tensor([h_distribution() for _ in range(numberOfJobs)])
    L = torch.tensor(np.random.randint(1, 1024, size=(numberOfJobs, )).tolist())
    W = torch.tensor(180 / numberOfMachines * 1000)
    P = torch.tensor(0.1)
    n = torch.tensor((10 ** (-174 / 10)) / 1000 * (180 / numberOfMachines * 1000))
    return (h, L, numberOfJobs, numberOfMachines, W, P, n)


class NOMAenv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict(
            {
                "Graph": spaces.Box(0, 1, shape=(numberOfJobs, numberOfJobs+numberOfMachines), dtype=np.int32),
                "h": spaces.Box(2e-12, 9e-10, shape=(numberOfJobs,), dtype=np.float32),
                "L": spaces.Box(1, 1024, shape=(numberOfJobs,), dtype=np.float32),
                "W": spaces.Box(0, 0.18, dtype=np.float32),
                "P": spaces.Box(0.1, 0.1, dtype=np.float32),
                "N": spaces.Box(0, 8e-16, dtype=np.float32),
            }
        )

        self.action_space = spaces.Discrete((numberOfJobs+numberOfMachines)*numberOfJobs)


    def action_to_tensor(self, numpyAction):
        if isinstance(numpyAction, torch.Tensor):
            return numpyAction
        rowPosition = numpyAction // (numberOfJobs+numberOfMachines)
        colPosition = numpyAction % (numberOfJobs+numberOfMachines)
        actionTensor = torch.zeros((numberOfJobs, numberOfJobs+numberOfMachines))
        actionTensor[rowPosition][colPosition] = 1
        return actionTensor


    def calculate_time_nodummy(self, Graph):
        G_tmp = Graph[:, 0:self.numberOfJobs]
        row = torch.sum(G_tmp, dim=1)
        col = torch.sum(G_tmp, dim=0)

        totalTime_OMA_fake = (1 - row - col) * self.T_list
        totalTime_NOMA_fake = torch.sum(G_tmp * self.T, dim=0)
        totalTime_fake = totalTime_OMA_fake + totalTime_NOMA_fake

        return torch.max(totalTime_fake @ (Graph[:, self.numberOfJobs: Graph.size()[1]]))


    def calculate_time_dummy(self, Graph):
        row = torch.sum(Graph, dim=1)
        totalTime_dummy = torch.sum((1 - row) * self.T_list)
        return torch.max(totalTime_dummy, self.calculate_time_nodummy(Graph))


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.h, self.L, self.numberOfJobs, self.numberOfMachines, self.W, self.P, self.n = sample_env()
        jobs = list(zip(self.h.flatten().tolist(), self.L.flatten().tolist()))
        self.jobList = torch.tensor(sorted(jobs, key=lambda x: x[0], reverse=True))

        self.G = torch.zeros((self.numberOfJobs, self.numberOfJobs + self.numberOfMachines))
        self.T = build_time_matrix(self.jobList, self.numberOfJobs, self.W, self.P, self.n)[0]
        self.T_list = build_time_matrix(self.jobList, self.numberOfJobs, self.W, self.P, self.n)[1]

        return (self.get_obs(), self.get_info())


    def step(self, Action):
        Action = self.action_to_tensor(Action)
        self.G += self.mask(self.G) * Action
        reward = self.reward(self.G)
        mask = self.mask(self.G)
        is_done = self.is_done()
        return (self.get_obs(), reward * (not is_done), is_done, False, {"action_mask": mask})


    def mask(self, Graph: torch.tensor):
        left = torch.ones((Graph.size()[0], Graph.size()[0]))
        right = torch.ones((Graph.size()[0], Graph.size()[1] - Graph.size()[0]))
        row = torch.sum(Graph, dim=1, keepdim=True)
        col = torch.sum(Graph, dim=0, keepdim=True)

        left = left - row - torch.t(row) - col[:, 0:Graph.size()[0]] - torch.t(col[:, 0:Graph.size()[0]])
        left = torch.where(left == 1, torch.tensor(1).float(), torch.tensor(0).float())
        left = left.triu(diagonal=1)
        right -= row

        return torch.cat((left, right), dim=1)


    def sample(self):
        mask = self.mask(self.G)
        indices = torch.nonzero(mask)

        if indices.numel() > 0:
            selected_index = random.choice(indices)
            index = selected_index[0] * mask.shape[1] + selected_index[1]
        else:
            raise ValueError("No available action")

        # sampleMatrix = torch.zeros_like(mask)
        # if indices.numel() > 0:
        #     selected_index = random.choice(indices)
        #     sampleMatrix[selected_index[0], selected_index[1]] = 1

        return index.clone().numpy()


    def get_obs(self):
        return {
                "Graph": self.G,
                "h": self.h,
                "L": self.L,
                "W": self.W,
                "P": self.P,
                "N": self.n
        }

    def get_info(self):
        return {
            "action_mask": self.mask(self.G)
        }


    def get_parameters(self):
        return (self.h, self.L, self.W, self.P, self.n)


    def reward(self, Graph):
        return -self.calculate_time_dummy(Graph)


    def is_done(self):
        return torch.sum(self.mask(self.G)) == 0


    def close(self):
        sys.exit()


if __name__ == "__main__":
    env = NOMAenv()
    env.reset(seed=42)
    for _ in range(20):
        action = env.sample()
        observation, reward, done, _, info = env.step(action)
        print(f"""observation = {observation}""")
        print(f"""reward = {reward}""")
        print(f"""done = {done}""")
        print(f"""info = {info}""")
        print("_____________________________")
        if done:
            env.reset()
