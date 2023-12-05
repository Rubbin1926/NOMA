from env import NOMAenv
from heteroGNN import heteroGNN
import torch
import time
start_time = time.time()


env = NOMAenv()
observation, info = env.reset(seed=42)
numberOfJobs = observation.shape[0]
numberOfMachines = observation.shape[1] - numberOfJobs

gnn = heteroGNN(numberOfJobs, numberOfMachines,
                job_output_features_number=3, machine_output_features_number=5)

DataSet = {}


def generate_mask_list(mask):
    mask_indices = torch.nonzero(mask)  # 获取mask中为1的元素的索引位置
    tensors = []

    for index in mask_indices:
        tensor = torch.zeros_like(mask)  # 创建和mask大小相同的全0矩阵
        tensor[index[0], index[1]] = 1  # 将对应位置的元素设置为1
        tensors.append(tensor)

    return tensors


def find(Graph, Dataset):
    mask = env.mask(Graph)

    if torch.sum(mask) == 0:
        Dataset[Graph] = (None, Graph, env.calculate_time_dummy(Graph))
        return Graph, env.calculate_time_dummy(Graph)

    if Graph in Dataset:
        return Dataset[Graph][1], Dataset[Graph][2]

    V_star = -1e16
    A_star = None
    G_star = None
    mask_list = generate_mask_list(mask)

    for action in mask_list:
        G_, V_ = find(Graph+action, Dataset)
        if V_ > V_star:
            V_star = V_
            A_star = action
            G_star = G_

    Dataset[Graph] = A_star, G_star, V_star
    return G_star, V_star



find(observation, DataSet)
print(DataSet)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序执行时间：{elapsed_time}秒")


