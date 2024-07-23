from env import NOMAenv, BATCH_SIZE
from policy import NOMAInitEmbedding, NOMAContext, NOMADynamicEmbedding

import torch
import torch.nn as nn

from rl4co.models.common.constructive.base import NoEncoder
from rl4co.utils.decoding import rollout, random_policy
from rl4co.models.zoo import AttentionModel, AttentionModelPolicy
from rl4co.models.rl import PPO
from rl4co.utils.trainer import RL4COTrainer

env = NOMAenv()
emb_dim = 64
policy = AttentionModelPolicy(env_name=env.name, # this is actually not needed since we are initializing the embeddings!
                              embed_dim=emb_dim,
                              init_embedding=NOMAInitEmbedding(emb_dim),
                              context_embedding=NOMAContext(emb_dim),
                              dynamic_embedding=NOMADynamicEmbedding(emb_dim)
)

model = AttentionModel(env,
                       policy=policy,
                       baseline='rollout',
                       train_data_size=128,
                       val_data_size=128)

# model = PPO(env, policy=policy, train_data_size=100, val_data_size=100)

# Greedy rollouts over untrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
td_init = env.reset(batch_size=BATCH_SIZE).to(device)
policy = model.policy.to(device)
# breakpoint()
# out = policy(td_init.clone(), env, phase="test", decode_type="sampling", return_actions=True)
# print(out)
# actions_untrained = out['actions'].cpu().detach()
# rewards_untrained = out['reward'].cpu().detach()
#
# for i in range(BATCH_SIZE):
#     print(f"Problem {i+1} | Cost: {-rewards_untrained[i]:.4f}")

trainer = RL4COTrainer(max_epochs=20, devices=1, log_every_n_steps=1)
trainer.fit(model)

# out = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)
# actions_trained = out['actions'].cpu().detach()
