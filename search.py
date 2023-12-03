from env import NOMAenv
from heteroGNN import heteroGNN
import torch


env = NOMAenv()
observation, info = env.reset(seed=42)
numberOfJobs = observation.shape[0]
numberOfMachines = observation.shape[1] - numberOfJobs

gnn = heteroGNN(numberOfJobs, numberOfMachines,
                job_output_features_number=3, machine_output_features_number=5)


def optimal_solution_and_value(Graph):
    mask = env.mask(Graph)
    if torch.sum(mask) == 0:
        return env.calculate_time_nodummy(Graph), Graph


    def generate_mask_list(mask):
        mask_indices = torch.nonzero(mask)  # 获取mask中为1的元素的索引位置
        tensors = []

        for index in mask_indices:
            tensor = torch.zeros_like(mask)  # 创建和mask大小相同的全0矩阵
            tensor[index[0], index[1]] = 1  # 将对应位置的元素设置为1
            tensors.append(tensor)

        return tensors

    mask_list = generate_mask_list(mask)

    # newGraph = Graph + mask_list[0]
    # return optimal_solution_and_value(newGraph)

    for action in mask_list:
        newGraph = Graph + action
        terminal_conv, Value, Possibility = gnn.forward(Graph, env.h, env.L, env.W, env.P, env.n)
        print(terminal_conv, Value, Possibility)
        print("____________")
        return optimal_solution_and_value(newGraph)






    # for mask in all_mask:
    #     newGraph  = Graph + mask
    #     optimal_time, optimal_Graph = optimal_solution_and_value(newGraph)

print(optimal_solution_and_value(observation))


