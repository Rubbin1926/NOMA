from mwmatching import maxWeightMatching
import heapq
import tensordict

import torch

from torch import nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.pylogger import get_pylogger
from rl4co.utils.ops import calculate_entropy
from typing import *
log = get_pylogger(__name__)

# def longest_processing_time_first(tasks, num_machines):
#     # 初始化机器的空闲时间和任务分配情况
#     machine_times = [0] * num_machines
#     machine_tasks = [[] for _ in range(num_machines)]
#     machine_heap = [(0, i) for i in range(num_machines)]
#     heapq.heapify(machine_heap)
#
#     for task_time in tasks:
#         # 从堆中获取最早空闲的机器
#         machine_idle_time, machine_idx = heapq.heappop(machine_heap)
#         # 更新机器的空闲时间
#         machine_idle_time += task_time
#         # 记录任务分配情况
#         machine_tasks[machine_idx].append(task_time)
#         # 将更新后的机器空闲时间和机器编号重新加入堆中
#         heapq.heappush(machine_heap, (machine_idle_time, machine_idx))
#
#     # 根据任务分配情况输出每个机器处理的任务
#     result = [machine_tasks[i] for i in range(num_machines)]
#
#     return result, max(machine_heap)[0]

def longest_processing_time_first(tasks_with_info, num_machines):
    # 初始化机器的空闲时间和任务分配情况
    machine_times = [0] * num_machines
    machine_tasks = [[] for _ in range(num_machines)]  # 存储（时间， job对）列表
    job_assignment = {}  # 记录每个job的machine分配
    machine_heap = [(0, i) for i in range(num_machines)]
    heapq.heapify(machine_heap)

    # 按时间降序排序任务
    sorted_tasks = sorted(tasks_with_info, key=lambda x: -x[0])

    for task_time, job_pair in sorted_tasks:
        # 获取最早空闲的机器
        machine_idle_time, machine_idx = heapq.heappop(machine_heap)
        new_idle_time = machine_idle_time + task_time

        # 记录任务到机器
        machine_tasks[machine_idx].append((task_time, job_pair))

        # 更新job_assignment
        i, j = job_pair
        if i == j:
            # 单独任务
            job_assignment[i] = machine_idx
        else:
            # 配对任务，两个job分配到同一个machine
            job_assignment[i] = machine_idx
            job_assignment[j] = machine_idx

        # 将机器放回堆中
        heapq.heappush(machine_heap, (new_idle_time, machine_idx))

    # 计算各machine的时间列表和最大时间
    machine_time_lists = []
    max_time = 0
    for m_idx in range(num_machines):
        times = [task[0] for task in machine_tasks[m_idx]]
        machine_time_lists.append(times)
        total = sum(times)
        if total > max_time:
            max_time = total

    return machine_time_lists, max_time, job_assignment


def tensor_to_list(tensor):
    result = []
    for batch in tensor:
        batch_list = []
        for i, row in enumerate(batch):
            for j, val in enumerate(row):
                if i < j:  # 仅处理上三角部分
                    batch_list.append((i, j, val.item()))
        result.append(batch_list)
    return result


def transform_list(input_list):
    result = []
    for inner_list in input_list:
        transformed_inner_list = [(index, index if value == -1 else value) for index, value in enumerate(inner_list)]
        result.append(transformed_inner_list)
    return result


def OMA_LPT(td: tensordict.TensorDict):
    taskAssignment, timeNeed = [], []

    batch_size = td.batch_size[0]
    T_list_batch = td["T_list"]

    for batch_id in range(batch_size):
        T_list = T_list_batch[batch_id].flatten().tolist()
        T_list.sort(reverse=True)

        num_machines = td["numberOfMachines"][batch_id].item()

        ret = longest_processing_time_first(T_list, num_machines)
        taskAssignment.append(ret[0])
        timeNeed.append(ret[1])

    return taskAssignment, timeNeed


# def NOMA_LPT(td: tensordict.TensorDict):
#     taskAssignment, timeNeed = [], []
#
#     batch_size = td.batch_size[0]
#     T_matrix_batch, max_job = td["T"], td["max_job"][0].item()
#
#     print(f"T_matrix_batch : {T_matrix_batch}")
#
#     T_matrix_negatives = T_matrix_batch.clone()
#     for i in range(batch_size):
#         for j in range(max_job):
#             T_matrix_negatives[i, j, j] = 0
#     T_matrix_negatives = 1 - T_matrix_negatives
#     list_for_matching = tensor_to_list(T_matrix_negatives)
#
#     print(f"list_for_matching : {list_for_matching}")
#
#     matching_output = []
#     for batch_id in range(batch_size):
#         matching_output.append(maxWeightMatching(list_for_matching[batch_id]))
#
#     matching_output = transform_list(matching_output)
#
#     print(f"matching_output : {matching_output}")
#
#     matching_time = []
#     for batch_id in range(batch_size):
#         batch_matching_time = []
#         for matching in matching_output[batch_id]:
#             batch_matching_time.append(T_matrix_batch[batch_id, matching[0], matching[1]].item())
#             print(T_matrix_batch[batch_id, matching[0], matching[1]])
#         batch_matching_time.sort(reverse=True)
#         print(f"batch_matching_time : {batch_matching_time}")
#         matching_time.append(batch_matching_time)
#
#     for batch_id in range(batch_size):
#         num_machines = td["numberOfMachines"][batch_id].item()
#         ret = longest_processing_time_first(matching_time[batch_id], num_machines)
#         taskAssignment.append(ret[0])
#         timeNeed.append(ret[1])
#
#     return taskAssignment, timeNeed, matching_output

def NOMA_LPT(td: tensordict.TensorDict):
    taskAssignment, timeNeed, job_assignments = [], [], []

    batch_size = td.batch_size[0]
    T_matrix_batch, max_job = td["T"], td["max_job"][0].item()

    T_matrix_negatives = T_matrix_batch.clone()
    for i in range(batch_size):
        for j in range(max_job):
            T_matrix_negatives[i, j, j] = 0
    T_matrix_negatives = 1 - T_matrix_negatives
    list_for_matching = tensor_to_list(T_matrix_negatives)

    matching_output = []
    for batch_id in range(batch_size):
        matching_output.append(maxWeightMatching(list_for_matching[batch_id]))

    matching_output = transform_list(matching_output)

    # 生成带job对的任务列表（修复重复配对问题）
    matching_time = []
    for batch_id in range(batch_size):
        batch_matching = []
        seen = set()
        for matching in matching_output[batch_id]:
            i, j = matching[0], matching[1]

            # 处理单独任务
            if i == j:
                if i not in seen:
                    time = T_matrix_batch[batch_id, i, j].item()
                    batch_matching.append((time, (i, j)))
                    seen.add(i)
            # 处理配对任务
            else:
                # 确保每个配对只处理一次（i < j）
                a, b = (i, j) if i < j else (j, i)
                if (a, b) not in seen:
                    time = T_matrix_batch[batch_id, a, b].item()
                    batch_matching.append((time, (a, b)))
                    seen.add((a, b))

        # 按时间降序排序
        batch_matching.sort(reverse=True, key=lambda x: x[0])
        matching_time.append(batch_matching)

    for batch_id in range(batch_size):
        num_machines = td["numberOfMachines"][batch_id].item()
        ret = longest_processing_time_first(matching_time[batch_id], num_machines)
        taskAssignment.append(ret[0])
        timeNeed.append(ret[1])
        job_assignments.append(ret[2])

    return taskAssignment, timeNeed, matching_output, job_assignments



def generate_action_matrix(td):
    numOfJobs = td["max_job"][0].item()
    numOfMachines = td["max_machine"][0].item()
    taskAssignment, timeNeed, matching_output, job_assignments = NOMA_LPT(td)

    def generate_graph(matching_output, job_assignments, numOfJobs, numOfMachines):
        num_cases = len(matching_output)
        graph = []

        for case_idx in range(num_cases):
            matrix = [[0] * (numOfJobs + numOfMachines) for _ in range(numOfJobs)]
            current_matching = matching_output[case_idx]
            current_assignment = job_assignments[case_idx]

            # Process left part (job-job matching)
            for pair in current_matching:
                i, j = pair
                if i < j:
                    matrix[i][j] = 1

            # Process right part (job-machine assignment)
            for job in range(numOfJobs):
                # Check if the job has any 1 in the left part
                has_left = any(matrix[job][:numOfJobs])
                if not has_left:
                    machine = current_assignment.get(job, 0)  # default to 0 if not found, though should exist
                    matrix[job][numOfJobs + machine] = 1

            graph.append(matrix)

        return torch.tensor(graph, dtype=torch.int32)


    Graph = generate_graph(matching_output, job_assignments, numOfJobs, numOfMachines)

    # print(Graph)


    def action_matrix(graph_tensor):
        batch_size = graph_tensor.size(0)
        num_jobs = graph_tensor.size(1)

        action_list = []
        for i in range(batch_size):
            sample = graph_tensor[i]
            flattened = sample.flatten()
            indices = torch.nonzero(flattened, as_tuple=False).squeeze(-1)

            # 验证每个样本的Job数量
            if indices.numel() != num_jobs:
                missing = num_jobs - indices.numel()
                raise ValueError(f"样本{i}缺失{missing}个Job的分配，请检查配对和分配逻辑")

            action_list.append(indices)

        return torch.stack(action_list, dim=0).to(torch.long)

    action_matrix = action_matrix(Graph)


    return action_matrix

def LPT_td(td):
    td.set("actions", generate_action_matrix(td))
    td.set("log_likelihood", torch.zeros(td.batch_size, dtype=torch.float32, device=td.device))

    return td









class NOMA_LPT_NET(nn.Module):
    def __init__(self):
        super(NOMA_LPT_NET, self).__init__()
        self.counter = 0
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, env, td: tensordict.TensorDict):
        device = td["Graph"].device
        if self.counter == 0:
            self.td = td.to(device)
            self.batch_size = self.td.batch_size[0]
            self.action_matrix = generate_action_matrix(td).to(device)
        # print("action_matrix : ", action_matrix)

        empty_tensor = torch.zeros_like(td["Graph"], device=device).reshape(self.batch_size, -1)
        fixed_score = torch.full((self.batch_size, 1), 1.0, device=device)

        index_tensor = self.action_matrix[:, self.counter].reshape(self.batch_size, -1)

        empty_tensor.scatter_(1, index_tensor, fixed_score)

        small_score = -0.5
        empty_tensor = empty_tensor.masked_fill(empty_tensor == 0, small_score)

        result = torch.log(self.softmax(empty_tensor))

        self.counter = (self.counter + 1) % self.action_matrix.shape[1]

        return result * 0, td["action_mask"], self.action_matrix


class LPTPolicy(nn.Module):
    def __init__(
        self,
        Net: nn.Module,
        temperature: float = 1.0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(LPTPolicy, self).__init__()

        if len(unused_kw) > 0:
            log.error(f"Found {len(unused_kw)} unused kwargs: {unused_kw}")

        self.Net = Net

        # Decoding strategies
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            calc_reward: Whether to calculate the reward
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_hidden: Whether to return the hidden state
            return_init_embeds: Whether to return the initial embeddings
            return_sum_log_likelihood: Whether to return the sum of the log likelihood
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            max_steps: Maximum number of decoding steps for sanity check to avoid infinite loops if envs are buggy (i.e. do not reach `done`)
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logits, mask, _ = self.Net.forward(env, td)

            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            # 此时td内的action被更新过了
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)
        # logprobs: 所有的actions对应的logprobs，shape = [batch_size, 走多少步]
        # actions: 所有的actions，shape = [batch_size, 走多少步]
        # 剩下两个没动

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))
        # 你终于算reward了，太感动了

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }




        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        # if return_hidden:
        #     outdict["hidden"] = hidden
        # if return_init_embeds:
        #     outdict["init_embeds"] = init_embeds

        return outdict




# [{1: 0, 0: 1, 3: 1, 2: 1, 4: 1}, {1: 0, 0: 1, 2: 1, 3: 1, 4: 1}]
# [[(0, 3), (1, 1), (2, 4), (3, 0), (4, 2)], [(0, 2), (1, 1), (2, 0), (3, 4), (4, 3)]]


# tensor([[[0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 0, 0, 0, 1],
#          [0, 0, 0, 0, 0, 0, 1]],
#
#         [[0, 0, 1, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 1],
#          [0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 0, 0, 0, 1]]], dtype=torch.int32)


