import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from heteroGNN import heteroGNN
from env import NOMAenv
import time
start_time = time.time()


env = NOMAenv()
observation, info = env.reset(seed=42)
numberOfJobs = observation.shape[0]
numberOfMachines = observation.shape[1] - numberOfJobs

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
        Dataset[Graph] = (torch.zeros_like(Graph), Graph, env.calculate_time_dummy(Graph))
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

# 3, 16, 136, 1677, 27751, 586018, 15226086

find(observation, DataSet)
print(DataSet)
print(f"""len = {len(DataSet)}""")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序执行时间：{elapsed_time}秒")

with open('dataset_4_2.pkl', 'wb') as file:
    pickle.dump(DataSet, file)



# 从文件中加载字典
with open('dataset_4_2.pkl', 'rb') as file:
    GNN_dataset = pickle.load(file)


class Read_Dataset(Dataset):
    def __init__(self, data_dict):
        self.keys = list(data_dict.keys())
        self.values = list(data_dict.values())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        x = self.keys[idx]
        y = self.values[idx][2]  # 取Value作为y
        return x, y


# 获取关键两个参数
# first_key = next(iter(GNN_dataset))
# numberOfJobs = first_key.shape[0]
# numberOfMachines = first_key.shape[1] - numberOfJobs

Train_dataset = Read_Dataset(GNN_dataset)
dataloader = DataLoader(Train_dataset, batch_size=8, shuffle=False)

gnn = heteroGNN(numberOfJobs, numberOfMachines,
                job_output_features_number=3, machine_output_features_number=5)
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.1)
loss_f = nn.MSELoss()

for i in range(100):
    for X, y in Train_dataset:
        pre_y = gnn.forward(X.type(torch.float32), env.h, env.L, env.W, env.P, env.n)[0]
        loss = loss_f(pre_y, y.type(torch.float32))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"""i = {i} loss = {loss}""")