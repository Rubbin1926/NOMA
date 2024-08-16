from env import NOMAenv, BATCH_SIZE
from env import numberOfJobs, numberOfMachines
from policy import NOMAInitEmbedding, NOMAContext, NOMADynamicEmbedding, MyCriticNetwork
from search import print_best_solution
import wandb
from lightning.pytorch.loggers import WandbLogger

import torch

from rl4co.utils.decoding import rollout, random_policy
from rl4co.models.zoo import AttentionModel, AttentionModelPolicy
from rl4co.models.zoo.symnco.policy import SymNCOPolicy
from rl4co.models.rl import PPO
from rl4co.utils.trainer import RL4COTrainer

wandb.login(key="a92a309a25837dfaeac912d8a533448c9bb7399a")
logger = WandbLogger(project="NOMA", config={"numberOfJobs": numberOfJobs,
                                             "numberOfMachines": numberOfMachines,
                                             "Policy": "AttentionModel", })
# logger = None

env = NOMAenv()
emb_dim = 128
policy = AttentionModelPolicy(env_name=env.name, # this is actually not needed since we are initializing the embeddings!
                              embed_dim=emb_dim,
                              init_embedding=NOMAInitEmbedding(emb_dim),
                              context_embedding=NOMAContext(emb_dim),
                              dynamic_embedding=NOMADynamicEmbedding(emb_dim)
)

# model = AttentionModel(env,
#                        policy=policy,
#                        baseline='rollout',
#                        train_data_size=8,
#                        val_data_size=8)

model = PPO(env,
            policy=policy,
            train_data_size=128,
            val_data_size=128,
            normalize_adv=False,
            ppo_epochs=3,
            clip_range=0.1,
            critic=MyCriticNetwork(embed_dim=emb_dim),
            critic_kwargs={"embed_dim": emb_dim})

# Greedy rollouts over untrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
td_init = env.reset(batch_size=BATCH_SIZE)
policy = model.policy.to(device)
# print(policy)


# out = policy(td_init.clone(), env, phase="test", decode_type="sampling", return_actions=True)
# print(out)
# actions_untrained = out['actions'].cpu().detach()
# rewards_untrained = out['reward'].cpu().detach()
#
# for i in range(BATCH_SIZE):
#     print(f"Problem {i+1} | Cost: {-rewards_untrained[i]:.4f}")

trainer = RL4COTrainer(max_epochs=20, devices=1, logger=logger, log_every_n_steps=1)
# breakpoint()
trainer.fit(model)

out = policy(td_init.clone(), env, phase="test", return_actions=True, return_init_embeds=False)
print(out)
actions = out['actions']
print(f"""after policy: {env.step_to_end_from_actions(td_init.clone(), actions)["Graph"]}""")
print(f"""best reward: {print_best_solution(td_init.clone())}""")

