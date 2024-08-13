from tensordict.tensordict import TensorDict
import torch
from env import NOMAenv, mask, reward_multiplicative_factor


def calculate_time_nodummy(td: TensorDict, numberOfJobs, numberOfMachines) -> torch.Tensor:
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


def calculate_time_dummy(td: TensorDict, numberOfJobs, numberOfMachines) -> torch.Tensor:
    Graph, T_list = td["Graph"], td["T_list"]
    Graph = Graph.reshape(-1, numberOfJobs, numberOfJobs + numberOfMachines)
    T_list = T_list.reshape(-1, 1, numberOfJobs)

    row = torch.sum(Graph, dim=-1).reshape_as(T_list)
    totalTime_dummy = torch.sum((1 - row) * T_list, dim=-1).flatten()

    ret, _ = torch.max(torch.stack((totalTime_dummy, calculate_time_nodummy(td, numberOfJobs, numberOfMachines))), dim=0)
    return reward_multiplicative_factor * ret


def generate_mask_list(Mask: torch.Tensor) -> list[torch.Tensor]:
    mask_indices = torch.nonzero(Mask)  # 获取mask中为1的元素的索引位置
    tensors = []

    for index in mask_indices:
        tensor = torch.zeros_like(Mask)  # 创建和mask大小相同的全0矩阵
        tensor[index[0], index[1]] = 1  # 将对应位置的元素设置为1
        tensors.append(tensor)

    return tensors


def find(td, Dataset):
    numberOfJobs = td['T'].shape[-1]
    Graph = td['Graph'].reshape(numberOfJobs, -1)
    numberOfMachines = Graph.shape[-1] - numberOfJobs
    Mask = mask(Graph).reshape_as(Graph)

    if torch.sum(Mask) == 0:
        td_tmp = TensorDict({"Graph": Graph, "T": td["T"], "T_list": td["T_list"]})
        Dataset[Graph] = (torch.zeros_like(Graph),
                          Graph,
                          calculate_time_dummy(td_tmp, numberOfJobs, numberOfMachines))
        return Graph, calculate_time_dummy(td_tmp, numberOfJobs, numberOfMachines)

    if Graph in Dataset:
        return Dataset[Graph][1], Dataset[Graph][2]

    V_star = -1e16
    A_star = None
    G_star = None
    mask_list = generate_mask_list(Mask)

    for action in mask_list:
        td_tmp = TensorDict({"Graph": Graph+action, "T": td["T"], "T_list": td["T_list"]})
        G_, V_ = find(td_tmp, Dataset)
        if V_ > V_star:
            V_star = V_
            A_star = action
            G_star = G_

    Dataset[Graph] = (A_star,
                      G_star,
                      V_star)
    return G_star, V_star


def find_best_solution(td: TensorDict) -> list[dict]:
    """
    :param td: TensorDict
    :return: list[dict], 字典中包含了每个batch中每种可能的图的最优解
    """
    tensor_dict_lst = [TensorDict({key: value[index] for key, value in td.items()}) for index in range(td.batch_size[0])]
    lst = []
    for td in tensor_dict_lst:
        dat = {}
        find(td, dat)
        lst.append(dat)
    return lst


if __name__ == '__main__':
    from env import BATCH_SIZE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 64

    env = NOMAenv()

    td = env.reset(batch_size=[BATCH_SIZE])
    lst = find_best_solution(td)
    print(lst)
