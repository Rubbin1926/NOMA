import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from heteroGNN import heteroGNN
from env import NOMAenv
import time
start_time = time.time()


env_train = NOMAenv()
env_test = NOMAenv()
observation, info = env_train.reset(seed=42)
numberOfJobs = observation.shape[0]
numberOfMachines = observation.shape[1] - numberOfJobs

trainDataSet = {}
testDataSet = {}


def generate_mask_list(mask):
    mask_indices = torch.nonzero(mask)  # 获取mask中为1的元素的索引位置
    tensors = []

    for index in mask_indices:
        tensor = torch.zeros_like(mask)  # 创建和mask大小相同的全0矩阵
        tensor[index[0], index[1]] = 1  # 将对应位置的元素设置为1
        tensors.append(tensor)

    return tensors


def find(Graph, Dataset, env):
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
        G_, V_ = find(Graph+action, Dataset, env)
        if V_ > V_star:
            V_star = V_
            A_star = action
            G_star = G_

    Dataset[Graph] = A_star, G_star, V_star
    return G_star, V_star

# 3, 16, 136, 1677, 27751, 586018, 15226086

find(observation, trainDataSet, env_train)
print(trainDataSet)
print(f"""len = {len(trainDataSet)}""")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序执行时间：{elapsed_time}秒")

# 生成test数据集
observation, info = env_test.reset(seed=42)
find(observation, testDataSet, env_test)


# with open('trainDataset.pkl', 'wb') as file:
#     pickle.dump(trainDataSet, file)
#
#
# # 从文件中加载字典
# with open('trainDataset.pkl', 'rb') as file:
#     trainDataSet = pickle.load(file)


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


def validate(model: heteroGNN, dataset: Read_Dataset, env: NOMAenv):
    cor, num = 0, dataset.__len__()
    for X, y in dataset:
        pred = model.forward(X.type(torch.float32), env.h, env.L, env.W, env.P, env.n)[0]
        if torch.abs(pred - y) / y < 0.1:
            cor += 1
    acc = cor / num
    return acc


# 获取关键两个参数
# first_key = next(iter(GNN_dataset))
# numberOfJobs = first_key.shape[0]
# numberOfMachines = first_key.shape[1] - numberOfJobs


Train_dataset = Read_Dataset(trainDataSet)
Test_dataset = Read_Dataset(testDataSet)
dataloader = DataLoader(Train_dataset, batch_size=64, shuffle=False)

gnn = heteroGNN(numberOfJobs, numberOfMachines,
                job_output_features_number=10, machine_output_features_number=10)
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.0001, weight_decay=1e-5)
loss_f = nn.MSELoss()

for i in range(100):
    gnn.train()
    for X, y in Train_dataset:
        pre_y = gnn.forward(X.type(torch.float32), env_train.h, env_train.L, env_train.W, env_train.P, env_train.n)[0]
        loss = loss_f(pre_y, y.type(torch.float32))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"""i = {i} loss = {loss}""")

    if i % 3 == 0:
        gnn.eval()
        acc_train = validate(gnn, Train_dataset, env_train)
        acc_test = validate(gnn, Test_dataset, env_test)
        print(f"After {i}, acc_train = {acc_train}")
        print(f"After {i}, acc_tset = {acc_test}")


 # 加层数
 # machine取max
