import torch
import math
import gym
from typing import Optional
import random
import numpy as np
import sys


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

    numberOfJobs = 4
    numberOfMachines = 2

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

class NOMAenv(gym.Env):
    #jobs:[[h,L],[],...,[]]
    def __init__(self):
        pass
        # jobs = list(zip(h, L))
        # self.jobList = sorted(jobs, key=lambda x: x[0], reverse=True)
        # self.numberOfMachines = numberOfMachines
        # self.numberOfJobs = len(self.jobList)
        # self.W, self.P, self.n = W, P, n
        #
        # self.G = torch.zeros((self.numberOfJobs, self.numberOfJobs + self.numberOfMachines))
        # self.T = build_time_matrix(self.jobList, self.numberOfJobs, W, P, n)[0]
        # self.T_list = build_time_matrix(self.jobList, self.numberOfJobs, W, P, n)[1]
        #
        # #test case
        # self.G = torch.tensor([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]).float()


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


    def step(self, Action):
        self.G += Action
        reward = self.reward(self.G)
        mask = self.mask(self.G)
        is_done = self.is_done()
        return (self.G, reward, is_done, False, {"action_mask": mask})


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

        #test case
        # self.G = torch.tensor([[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]).float()

        return (self.G, {"action_mask": self.mask(self.G)})


    def sample(self):
        mask = self.mask(self.G)
        sampleMatrix = torch.zeros_like(mask)  # 创建和a相同size的全0张量b
        indices = torch.nonzero(mask)

        if indices.numel() > 0:
            selected_index = random.choice(indices)
            sampleMatrix[selected_index[0], selected_index[1]] = 1

        return sampleMatrix



    def reward(self, Graph):
        return -self.calculate_time_dummy(Graph)


    def is_done(self):
        if torch.sum(self.mask(self.G)):
            return False
        else:
            return True


    def close(self):
        sys.exit()



# tmp test
# aaa = NOMAenv()
# aaa.reset()
# t_d = aaa.calculate_time_dummy(Graph=torch.tensor([[0, 1, 0, 0, 0, 0],
#                                                    [0, 0, 0, 0, 1, 0],
#                                                    [0, 0, 0, 0, 0, 1],
#                                                    [0, 0, 0, 0, 0, 1]]).float())
# t_nd = aaa.calculate_time_nodummy(Graph=torch.tensor([[0, 1, 0, 0, 0, 0],
#                                                       [0, 0, 0, 0, 1, 0],
#                                                       [0, 0, 0, 0, 0, 1],
#                                                       [0, 0, 0, 0, 0, 1]]).float())
# ma = aaa.mask(Graph=torch.tensor([[0, 0, 0, 0, 0, 0],
#                                   [0, 0, 0, 0, 0, 0],
#                                   [0, 0, 0, 1, 0, 0],
#                                   [0, 0, 0, 0, 0, 0]]).float())
# re = aaa.reward(Graph=torch.tensor([[0, 1, 0, 0, 0, 0],
#                                     [0, 0, 0, 0, 1, 0],
#                                     [0, 0, 0, 0, 0, 1],
#                                     [0, 0, 0, 0, 0, 1]]).float())
# action = aaa.step(Action=torch.tensor([[0, 0, 0, 0, 0, 0],
#                                          [0, 0, 0, 0, 0, 0],
#                                          [0, 0, 0, 0, 0, 0],
#                                          [0, 0, 0, 0, 0, 0]]).float())
# print(f"""time_dummy = {t_d}""")
# print(f"""time_nodummy = {t_nd}""")
# print(f"""mask = {ma}""")
# print(f"""reward = {re}""")
# print(f"""action = {action}""")
