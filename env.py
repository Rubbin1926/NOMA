import torch
import math
from typing import Optional
import random
import numpy as np
import sys
import gymnasium as gym
from gymnasium import spaces


numberOfJobs = 2
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
    # numberOfJobs = random.randint(8, 90)
    # numberOfMachines = random.randint(2, 12)

    """Just Int For Test"""
    # h = np.abs(np.random.normal(10, 1, numberOfJobs)).tolist()
    # L = np.abs(np.random.normal(100, 10, numberOfJobs)).tolist()
    # W = random.randint(1, 5)
    # P = random.randint(1, 5)
    # n = random.randint(1, 5)

    """Parameters in paper"""
    def h_distribution():
        d = random.uniform(0.1, 0.5)
        tmp0 = (128.1 + 37.6 * math.log(d, 10)) / 10
        tmp1 = 10 ** tmp0
        _h = 1 / tmp1
        return _h

    h = [h_distribution() for _ in range(numberOfJobs)]
    L = np.random.randint(1, 1024, size=(numberOfJobs, )).tolist()
    W = 180 / numberOfMachines * 1000
    P = 0.1
    n = (10 ** (-174 / 10)) / 1000 * (180 / numberOfMachines * 1000)
    return (h, L, numberOfJobs, numberOfMachines, W, P, n)


def ppoActiontoGraph(ppoAction: int):
    action = torch.zeros((numberOfJobs, numberOfJobs+numberOfMachines))
    if ppoAction < (numberOfJobs*(numberOfJobs-1)//2):
        number = ppoAction + 1
        n = numberOfJobs - 1
        count = 0
        while number > 0:
            number -= n
            count += n
            n -= 1
        row = (numberOfJobs - n - 1) - 1
        col = number - 1 - numberOfMachines
    else:
        number = ppoAction - (numberOfJobs*(numberOfJobs-1)//2)
        row = number // 2
        col = number % 2 + numberOfJobs

    action[row][col] = 1
    return action



class NOMAenv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict(
            {
                "Graph": spaces.Box(0, 1, shape=(numberOfJobs, numberOfJobs+numberOfMachines), dtype=int),
                "h": spaces.Box(2e-12, 9e-10, shape=(numberOfJobs,), dtype=float),
                "L": spaces.Box(1, 1024, shape=(numberOfJobs,), dtype=float),
                "W": spaces.Box(0, 0.18, dtype=float),
                "P": spaces.Box(0.1, 0.1, dtype=float),
                "N": spaces.Box(0, 8e-16, dtype=float),
            }
        )

        self.action_space = spaces.Discrete((numberOfJobs*(numberOfJobs-1)//2) + (numberOfJobs*numberOfMachines))


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
        jobs = list(zip(self.h, self.L))
        self.jobList = sorted(jobs, key=lambda x: x[0], reverse=True)

        self.G = torch.zeros((self.numberOfJobs, self.numberOfJobs + self.numberOfMachines))
        self.T = build_time_matrix(self.jobList, self.numberOfJobs, self.W, self.P, self.n)[0]
        self.T_list = build_time_matrix(self.jobList, self.numberOfJobs, self.W, self.P, self.n)[1]

        return (self.get_obs(), self.get_info())


    def step(self, Action):
        self.G += Action
        reward = self.reward(self.G)
        mask = self.mask(self.G)
        is_done = self.is_done()
        return (self.get_obs(), reward, is_done, False, {"action_mask": mask})


    def mask(self, Graph: torch.tensor):
        left = torch.ones((Graph.size()[0], Graph.size()[0]))
        right = torch.ones((Graph.size()[0], Graph.size()[1]-Graph.size()[0]))
        row = torch.sum(Graph, dim=1, keepdim=True)
        col = torch.sum(Graph, dim=0, keepdim=True)

        left = left - row - torch.t(row) - col[:, 0:Graph.size()[0]] - torch.t(col[:, 0:Graph.size()[0]])
        left = torch.where(left == 1, torch.tensor(1).float(), torch.tensor(0).float())
        left = left.triu(diagonal=1)
        right -= row

        return torch.cat((left, right), dim=1)


    def sample(self):
        mask = self.mask(self.G)
        sampleMatrix = torch.zeros_like(mask)  # 创建和a相同size的全0张量b
        indices = torch.nonzero(mask)

        if indices.numel() > 0:
            selected_index = random.choice(indices)
            sampleMatrix[selected_index[0], selected_index[1]] = 1

        return sampleMatrix


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
        if torch.sum(self.mask(self.G)):
            return False
        else:
            return True


    def close(self):
        sys.exit()


if __name__ == "__main__":
    env = NOMAenv()
    env.reset(seed=42)
    print(env.get_obs())
