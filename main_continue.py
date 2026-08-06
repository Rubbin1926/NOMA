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
from rl4co.utils.trainer import RL4COTrainer

from myPolicy import *
from myPPO import myPPO
from mySupervisedLearning import mySupervisedLearning
from LPT import *


# wandb.login(key="a92a309a25837dfaeac912d8a533448c9bb7399a")
# logger = WandbLogger(project="NOMA", config={"网络更改": "entropy_lambda=0, ppo_epochs=5, quantile=0.7, emb_dim=256, batch_size=64, lr=1e-6", "环境": "OMA转NOMA"})
logger = None


env = NOMAenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

last_checkpoint_file = "./checkpoints/sl/last.ckpt"
model = myPPO.load_from_checkpoint(last_checkpoint_file, strict=True).to(device)
model.env = env

policy = model.policy.to(device)

# print(model)
# print(model.env)

checkpoint_callback = [ModelCheckpoint(dirpath="./checkpoints/continue",
                                       filename="aaa",  # save as {epoch}-{step}.ckpt
                                       save_top_k=1,
                                       save_last=True,  # save the last model
                                       monitor="val/reward",  # monitor validation reward
                                       mode="max")]  # maximize validation reward
# checkpoint_callback = None


trainer = RL4COTrainer(max_epochs=3, devices=1, logger=logger, log_every_n_steps=1, callbacks=checkpoint_callback)
trainer.fit(model)