from env import NOMAenv, OMAenv, BATCH_SIZE
from policy import NOMAInitEmbedding, NOMAContext, NOMADynamicEmbedding, MyCriticNetwork
from search import print_best_solution
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary

import torch
import pickle

from rl4co.utils.decoding import rollout, random_policy
from rl4co.models.zoo import AttentionModel, AttentionModelPolicy
from rl4co.models.zoo.symnco.policy import SymNCOPolicy
from rl4co.models.rl import PPO
from rl4co.utils.trainer import RL4COTrainer

from myPolicy import *
    
wandb.login(key="a92a309a25837dfaeac912d8a533448c9bb7399a")
logger = WandbLogger(project="NOMA", config={"Env": "OMA", })
# logger = None

env = OMAenv()
emb_dim = 256
# policy = AttentionModelPolicy(env_name=env.name, # this is actually not needed since we are initializing the embeddings!
#                               embed_dim=emb_dim,
#                               init_embedding=NOMAInitEmbedding(emb_dim),
#                               context_embedding=NOMAContext(emb_dim),
#                               dynamic_embedding=NOMADynamicEmbedding(emb_dim),
#                               check_nan=True,)

policy = GNNPolicy(NOMANet=NOMANet(embed_dim=emb_dim))

model = PPO(env,
            policy=policy,

            mini_batch_size=1.0,
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
            optimizer_kwargs={"lr": 1e-7},)

# Greedy rollouts over untrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = model.policy.to(device)
# print(policy)

checkpoint_callback = [ModelCheckpoint(dirpath="checkpoints/OMA_my_policy",
                                       filename=None,  # save as {epoch}-{step}.ckpt
                                       save_top_k=1,
                                       save_last=True,  # save the last model
                                       monitor="val/reward",  # monitor validation reward
                                       mode="max")]  # maximize validation reward
# checkpoint_callback = None

trainer = RL4COTrainer(max_epochs=200, devices=1, logger=logger, log_every_n_steps=1, callbacks=checkpoint_callback)
trainer.fit(model)

# td_init = env.reset(batch_size=BATCH_SIZE)
# out = policy(td_init.clone(), env, phase="test", return_actions=True, return_init_embeds=False)
# print(out)
# actions = out['actions']
# print(actions)
# print(f"""after policy: {env.step_to_end_from_actions(td_init.clone(), actions)["Graph"]}""")
# print(f"""best reward: {print_best_solution(td_init.clone())}""")

