from env import NOMAenv, OMAenv, BATCH_SIZE
from policy import NOMAInitEmbedding, NOMAContext, NOMADynamicEmbedding, MyCriticNetwork
import torch
import pickle
import numpy as np
from rl4co.models.zoo import AttentionModelPolicy
from rl4co.models.rl import PPO
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.utils.decoding import rollout

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
env = NOMAenv().to(device)

# Load the last checkpoint file
last_checkpoint_file = "./checkpoints/NOMA_my_policy/p337/last.ckpt"
td_init = load_npz_to_tensordict("./tensordict/NOMA/td_20.npz").to(device)
# td_init = env.reset(batch_size=[BATCH_SIZE]).to(device)

# Load the model from the last checkpoint
model = PPO.load_from_checkpoint(last_checkpoint_file, strict=True).to(device)
policy = model.policy.to(device)

# Calculate reward using the model
out = policy(td_init.clone(), env, phase="test", return_actions=True)
reward = env.convert_to_true_reward(out["reward"]).mean().item()
print(reward)
actions = out["actions"]
print(env.step_to_end_from_actions(td_init.clone(), actions)["Graph"][0])
# print(out["reward"][0]*0.0021-0.0108)
print(env.step_to_end_from_actions(td_init.clone(), actions)["Graph"][1])
# print(out["reward"][1]*0.0021-0.0108)
print(env.step_to_end_from_actions(td_init.clone(), actions)["Graph"][2])
# print(out["reward"][2]*0.0021-0.0108)

# with open('reward/OMA/4_40/20_last_reward.pkl', 'wb') as f:
#     pickle.dump(reward, f)



# def my_random_policy(td):
#     action_mask = td["action_mask"].float()
#     actions = []
#     for mask in action_mask:
#         if mask.sum() == 0:
#             actions.append(-1)
#         else:
#             action = torch.multinomial(mask, 1).squeeze(-1)
#             actions.append(action.item())
#     td.set("action", torch.tensor(actions))
#     return td

# random_reward_list = []
# random_reward, td, actions = rollout(env, td_init.clone(), my_random_policy)
# print("__________________")
# print(td["Graph"][0])
# print(random_reward[0])
# print(td["Graph"][1])
# print(random_reward[1])
# print(td["Graph"][2])
# print(random_reward[2])
# random_reward = random_reward.tolist()
# random_reward_list.append(random_reward)

# with open('reward/OMA/4_40/20_random_reward.pkl', 'wb') as f:
#     pickle.dump(random_reward_list, f)