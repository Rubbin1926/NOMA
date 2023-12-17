import torch
import math
import gym
from typing import Optional
import random
import numpy as np
import sys
import math


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


def sample_env(batch_size):
    numberOfJobs = 18
    numberOfMachines = 4

    """Just Int For Test"""
    # h = np.abs(np.random.normal(10, 1, (batch_size, numberOfJobs))).tolist()
    # L = np.abs(np.random.normal(100, 10, (batch_size, numberOfJobs))).tolist()
    # W = np.random.randint(1, 5, size=(batch_size,)).tolist()
    # P = np.random.randint(1, 5, size=(batch_size,)).tolist()
    # n = np.random.randint(1, 5, size=(batch_size,)).tolist()

    """Parameters in paper"""
    def h_distribution():
        d = random.uniform(0.1, 0.5)
        tmp0 = (128.1 + 37.6 * math.log(d, 10)) / 10
        tmp1 = 10 ** tmp0
        _h = 1 / tmp1
        return _h

    h = [[h_distribution() for _ in range(numberOfJobs)] for _ in range(batch_size)]
    L = np.random.randint(1, 1024, size=(batch_size, numberOfJobs)).tolist()
    W = [180 / numberOfMachines * 1000] * batch_size
    P = [0.1] * batch_size
    n = [(10 ** (-174 / 10)) / 1000 * (180 / numberOfMachines * 1000)] * batch_size

    return (h, L, numberOfJobs, numberOfMachines, W, P, n)


class NOMAenv(gym.Env):
    #jobs:[[h,L],[],...,[]]
    def __init__(self, batch_size):
        self.batch_size = batch_size


    def calculate_time_nodummy(self, Graph):
        G_tmp = Graph[:, :, 0:self.numberOfJobs]
        row = torch.transpose(torch.sum(G_tmp, dim=2, keepdim=True), 1, 2)
        col = torch.sum(G_tmp, dim=1, keepdim=True)

        totalTime_OMA_fake = (1 - row - col) * self.T_list
        totalTime_NOMA_fake = torch.sum(G_tmp * self.T, dim=1, keepdim=True)
        totalTime_fake = totalTime_OMA_fake + totalTime_NOMA_fake

        Time_b_1_n = torch.bmm(totalTime_fake, (Graph[:, :, self.numberOfJobs: Graph.size()[2]]))
        max_values, _ = torch.max(Time_b_1_n, dim=2)

        return torch.squeeze(max_values, dim=1)


    def calculate_time_dummy(self, Graph):
        row = torch.transpose(torch.sum(Graph, dim=2, keepdim=True), 1, 2)
        totalTime_dummy = torch.sum((1 - row) * self.T_list, dim=2).view(-1)
        return torch.max(totalTime_dummy, self.calculate_time_nodummy(Graph))


    def mask(self, Graph):
        left = torch.ones((self.batch_size, self.numberOfJobs, self.numberOfJobs))
        right = torch.ones((self.batch_size, self.numberOfJobs, self.numberOfMachines))
        row = torch.sum(Graph, dim=2, keepdim=True)
        col = torch.sum(Graph, dim=1, keepdim=True)

        left = left - row - torch.transpose(row, 1, 2) - col[:, :, 0:self.numberOfJobs] - torch.transpose(col[:, :, 0:self.numberOfJobs], 1, 2)
        left = torch.where(left == 1, torch.tensor(1).float(), torch.tensor(0).float())
        left = left.triu(diagonal=1)
        right -= row

        return torch.cat((left, right), dim=2)


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
        h, L, self.numberOfJobs, self.numberOfMachines, W, P, n = sample_env(self.batch_size)

        combined_data = []
        for i in range(self.batch_size):
            combined_data.append(np.column_stack((h[i], L[i])).flatten())
        jobs = np.array(combined_data).reshape(self.batch_size, self.numberOfJobs, -1).tolist()

        self.jobMatrix = [sorted(sublist, key=lambda x: x[1], reverse=True) for sublist in jobs]

        self.G = torch.zeros((self.batch_size, self.numberOfJobs, self.numberOfJobs + self.numberOfMachines))

        self.T = torch.stack([build_time_matrix(self.jobMatrix[i], self.numberOfJobs, W[i], P[i], n[i])[0] for i in range(self.batch_size)])
        self.T_list = torch.stack([build_time_matrix(self.jobMatrix[i], self.numberOfJobs, W[i], P[i], n[i])[1] for i in range(self.batch_size)])

        return (self.G, {"action_mask": self.mask(self.G)})


    def sample(self):
        mask = self.mask(self.G)
        sampleMatrix = torch.zeros_like(mask)  # 创建和mask相同size的全0张量
        batch_size = mask.size(0)

        for i in range(batch_size):
            indices = torch.nonzero(mask[i])  # 获取第i个batch的非零元素索引
            if indices.numel() > 0:
                selected_index = random.choice(indices)
                sampleMatrix[i, selected_index[0], selected_index[1]] = 1  # 在第i个batch中设置相应位置为1

        return sampleMatrix


    def reward(self, Graph):
        return -self.calculate_time_dummy(Graph)


    def is_done(self):
        return not torch.sum(self.mask(self.G))


    def close(self):
        sys.exit()
