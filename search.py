import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from GNN import GraphNN
from env import NOMAenv
import wandb
import time

wandb.login(key="a92a309a25837dfaeac912d8a533448c9bb7399a")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()


def mask_f(Graph: torch.tensor):
    left = torch.ones((Graph.size()[0], Graph.size()[0])).to(device)
    right = torch.ones((Graph.size()[0], Graph.size()[1] - Graph.size()[0])).to(device)
    row = torch.sum(Graph, dim=1, keepdim=True)
    col = torch.sum(Graph, dim=0, keepdim=True)

    left = left - row - torch.t(row) - col[:, 0:Graph.size()[0]] - torch.t(col[:, 0:Graph.size()[0]])
    left = torch.where(left == 1, torch.tensor(1).float(), torch.tensor(0).float())
    left = left.triu(diagonal=1)
    right -= row

    return torch.cat((left, right), dim=1)


def generate_mask_list(mask):
    mask_indices = torch.nonzero(mask).to(device)  # 获取mask中为1的元素的索引位置
    tensors = []

    for index in mask_indices:
        tensor = torch.zeros_like(mask).to(device)  # 创建和mask大小相同的全0矩阵
        tensor[index[0], index[1]] = 1  # 将对应位置的元素设置为1
        tensors.append(tensor)

    return tensors


def find(Graph, Dataset, env):
    Graph = Graph.to(device)
    mask = mask_f(Graph)
    parameters = tuple(para.to(device) for para in env.get_parameters())

    if torch.sum(mask) == 0:
        ###注意这里将时间取log了###
        Dataset[Graph] = (torch.zeros_like(Graph).to(device),
                          Graph,
                          torch.log(env.calculate_time_dummy(Graph.to('cpu'))).to(device),
                          parameters)
        return Graph, env.calculate_time_dummy(Graph.to('cpu')).to(device)

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

    ###注意这里将时间取log了###
    Dataset[Graph] = (A_star,
                      G_star,
                      torch.log(V_star).to(device),
                      parameters)
    return G_star, V_star

# 3, 16, 136, 1677, 27751, 586018, 15226086


class Read_Dataset(Dataset):
    def __init__(self, data_dict: dict):
        self.keys = list(data_dict.keys())
        self.values = list(data_dict.values())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        x = self.keys[idx].to(device)
        y = self.values[idx][2].to(device)  # 取Value作为y
        parameters = tuple(para.to(device) for para in self.values[idx][3])
        return x, y, parameters


def validate(model: GraphNN, dataset: Read_Dataset):
    cor, num = 0, dataset.__len__()
    for X, y, para in dataset:
        pred = model.forward(X.type(torch.float32), para[0], para[1], para[2], para[3], para[4])[0]
        if torch.abs((pred - y) / y) < 0.2:
            cor += 1
    print(f"""cor = {cor}""")
    print(f"""num = {num}""")
    acc = cor / num
    return acc


def main():
    # 生成test数据集
    env_test = NOMAenv()
    testDataSet = {}
    observation, info = env_test.reset(seed=42)
    find(observation["Graph"], testDataSet, env_test)
    Test_dataset = Read_Dataset(testDataSet)

    # 获取重要参数
    numberOfJobs = observation["Graph"].shape[0]
    numberOfMachines = observation["Graph"].shape[1] - numberOfJobs

    train_datasets = []
    for i in range(5):
        tmpDataSet = {}
        env_train = NOMAenv()
        observation, info = env_train.reset(seed=42)
        find(observation["Graph"], tmpDataSet, env_train)
        tmp_Train_dataset = Read_Dataset(tmpDataSet)
        train_datasets.append(tmp_Train_dataset)

    Train_dataset = ConcatDataset(train_datasets)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序执行时间：{elapsed_time}秒")

    # dataloader = DataLoader(Train_dataset, batch_size=64, shuffle=False)


    lr = 0.001
    epochs = 21
    gnn = GraphNN()
    optimizer = torch.optim.Adam(gnn.parameters(), lr=lr, weight_decay=1e-6)
    loss_f = nn.MSELoss()
    run = wandb.init(
        # Set the project where this run will be logged
        project="NOMA",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": epochs,
        },
    )

    for i in range(epochs):
        gnn.train()
        for X, y, para in Train_dataset:
            pre_y = gnn.forward(X.type(torch.float32).to(device), para[0], para[1], para[2], para[3], para[4])[0]
            loss = loss_f(pre_y, y.type(torch.float32).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"""i = {i} loss = {loss}""")

        gnn.eval()
        acc_train = validate(gnn, Train_dataset)
        acc_test = validate(gnn, Test_dataset)
        print(f"### After {i}, acc_train = {acc_train}")
        print(f"### After {i}, acc_test = {acc_test}")

        wandb.log({"acc_train": acc_train, "acc_test": acc_test, "loss": loss})

    # save the model
    torch.save(gnn.state_dict(), 'gnn.pt')


if __name__ == "__main__":
    main()

