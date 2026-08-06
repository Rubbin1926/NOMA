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
from myPPO import myPPO
from mySupervisedLearning import mySupervisedLearning
from LPT import *

# vim ~/miniconda3/envs/NOMA/lib/python3.9/site-packages/rl4co/models/rl/ppo/ppo.py

# def load(path):
#     with open(path, 'rb') as f:
#         loaded_list = pickle.load(f)
#         return loaded_list
# lst = load("./record/lyu.pkl")
# lst = []
# with open("./record/lyu.pkl", 'wb') as f:
#     pickle.dump(lst, f)
    
# wandb.login(key="") #No No No!
# logger = WandbLogger(project="NOMA", name="p346 参数参考matlab代码2 没有log(reward)", config={"网络更改": "Net_loop=1 GNN_loop=2, vf_lambda=无用, entropy_lambda=0, ppo_epochs=5, quantile=0.7, emb_dim=128, batch_size=128, lr=1e-6, factor=500, reward_add_eps=无用1e-3, max_grad_norm=1.0, logit_multiplier=10, normalize_adv=False",})
logger = None
# logger = WandbLogger(project="NOMA", name="p348 sl转rl过程测试", config={})

env = NOMAenv()
emb_dim = 256
# policy = AttentionModelPolicy(env_name=env.name, # this is actually not needed since we are initializing the embeddings!
#                               embed_dim=emb_dim,
#                               init_embedding=NOMAInitEmbedding(emb_dim),
#                               context_embedding=NOMAContext(emb_dim),
#                               dynamic_embedding=NOMADynamicEmbedding(emb_dim),
#                               check_nan=True,)

policy = GNNPolicy(NOMANet=NOMANet(embed_dim=emb_dim, logit_multiplier=10))

# model = myPPO(env,
#             policy=policy,
#
#             mini_batch_size=1.0,
#             batch_size=128,
#             val_batch_size=32,
#             test_batch_size=32,
#
#             train_data_size=256,
#             val_data_size=32,
#             test_data_size=32,
#
#             normalize_adv=False,
#             ppo_epochs=5,
#             clip_range=0.2,
#             entropy_lambda=0.0,
#             vf_lambda=0.3,
#             quantile=0.7,
#             critic=MyCriticNetwork(embed_dim=emb_dim),
#             critic_kwargs={"embed_dim": emb_dim},
#             optimizer_kwargs={"lr": 1e-6},
#             max_grad_norm=0.5,
#             print_grads=True)

# model = myPPO(env,
#             policy=policy,
#
#             mini_batch_size=1.0,
#             batch_size=32,
#             val_batch_size=32,
#             test_batch_size=32,
#
#             train_data_size=256,
#             val_data_size=32,
#             test_data_size=32,
#
#             normalize_adv=False,
#             ppo_epochs=5,
#             clip_range=0.2,
#             entropy_lambda=0.0,
#             vf_lambda=0.3,
#             quantile=0.7,
#             critic=MyCriticNetwork(embed_dim=emb_dim),
#             critic_kwargs={"embed_dim": emb_dim},
#             optimizer_kwargs={"lr": 1e-6},
#             max_grad_norm=0.5,
#             print_grads=True)

# my_policy = LPTPolicy(Net=NOMA_LPT_NET())
model = mySupervisedLearning(env,
            policy=policy,

            mini_batch_size=1.0,
            batch_size=32,
            val_batch_size=32,
            test_batch_size=32,

            train_data_size=256,
            val_data_size=32,
            test_data_size=32,

            normalize_adv=False,
            ppo_epochs=5,
            clip_range=0.2,
            entropy_lambda=0.0,
            vf_lambda=0.3,
            quantile=0.7,
            critic=MyCriticNetwork(embed_dim=emb_dim),
            critic_kwargs={"embed_dim": emb_dim},
            optimizer_kwargs={"lr": 1e-6},
            max_grad_norm=0.5,
            print_grads=True)

# Greedy rollouts over untrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = model.policy.to(device)
# print(policy)

# checkpoint_callback = [ModelCheckpoint(dirpath="./checkpoints/rl",
#                                        filename=None,  # save as {epoch}-{step}.ckpt
#                                        save_top_k=0,
#                                        save_last=True,  # save the last model
#                                        monitor="val/reward",  # monitor validation reward
#                                        mode="max")]  # maximize validation reward
checkpoint_callback = None

trainer = RL4COTrainer(max_epochs=50, devices=1, logger=logger, log_every_n_steps=1, callbacks=checkpoint_callback)
trainer.fit(model)

# td_init = env.reset(batch_size=BATCH_SIZE)
# out = policy(td_init.clone(), env, phase="test", return_actions=True, return_init_embeds=False)
# print(out)
# actions = out['actions']
# print(actions)
# print(f"""after policy: {env.step_to_end_from_actions(td_init.clone(), actions)["Graph"]}""")
# print(f"""best reward: {print_best_solution(td_init.clone())}""")

