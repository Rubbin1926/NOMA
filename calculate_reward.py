from env import NOMAenv, OMAenv, BATCH_SIZE
from policy import NOMAInitEmbedding, NOMAContext, NOMADynamicEmbedding, MyCriticNetwork

import torch
import os
import pickle
import re

from rl4co.utils.decoding import rollout, random_policy
from rl4co.models.zoo import AttentionModel, AttentionModelPolicy
from rl4co.models.rl import PPO
from rl4co.data.utils import save_tensordict_to_npz, load_npz_to_tensordict

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
env = OMAenv().to(device)
emb_dim = 256
policy = AttentionModelPolicy(env_name=env.name,
                              embed_dim=emb_dim,
                              init_embedding=NOMAInitEmbedding(emb_dim),
                              context_embedding=NOMAContext(emb_dim),
                              dynamic_embedding=NOMADynamicEmbedding(emb_dim),
                              check_nan=True,).to(device)
model = PPO(env,
            policy=policy,
            batch_size=16,
            val_batch_size=16,
            test_batch_size=16,
            train_data_size=128,
            val_data_size=64,
            test_data_size=64,
            normalize_adv=False,
            ppo_epochs=3,
            clip_range=0.2,
            critic=MyCriticNetwork(embed_dim=emb_dim),
            critic_kwargs={"embed_dim": emb_dim},
            optimizer_kwargs={"lr": 1e-5},).to(device)


# Function to extract step number from filename
def extract_step(filename):
    match = re.search(r'step=(\d+)\.ckpt', filename)
    return int(match.group(1)) if match else 0

# Function to check if step is above the threshold
def filter_by_step(filename, min_step):
    return extract_step(filename) >= min_step

checkpoint_dir = "checkpoints/OMA/train_60_80"
min_step_threshold = 1  # Set your minimum step value here

checkpoint_files = sorted(
    [file for file in os.listdir(checkpoint_dir) if file.endswith(".ckpt") and filter_by_step(file, min_step_threshold)],
    key=extract_step
)

# for file in checkpoint_files:
#     print(file)

reward_lst = []
td_init = load_npz_to_tensordict("td_80_90.npz").to(device)
# td_init = env.reset(batch_size=[1]).to(device)

def my_random_policy(td):
    """Helper function to select a random action from available actions or None if no action is available"""
    action_mask = td["action_mask"].float()
    actions = []
    for mask in action_mask:
        if mask.sum() == 0:
            actions.append(-1)
        else:
            action = torch.multinomial(mask, 1).squeeze(-1)
            actions.append(action.item())
    td.set("action", torch.tensor(actions))
    return td

random_reward, _, _ = rollout(env, td_init.clone(), my_random_policy)
random_reward = random_reward.mean().item()
reward_lst.append(random_reward)

for checkpoint_file in checkpoint_files:
    print(checkpoint_file)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    new_model_checkpoint = PPO.load_from_checkpoint(checkpoint_path, strict=True).to(device)
    policy_new = new_model_checkpoint.policy.to(device)
    out = policy_new(td_init.clone(), env, phase="test", return_actions=False)
    reward = out["reward"].mean().item()
    reward_lst.append(reward)
    del new_model_checkpoint

# 将list保存到文件中
with open('reward/OMA/60_80/80-90 jobs.pkl', 'wb') as f:
    pickle.dump(reward_lst, f)

