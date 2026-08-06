import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from torch.utils.data import DataLoader

from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.models.rl.common.critic import CriticNetwork, create_critic_from_actor

from LPT import generate_action_matrix

import abc

from functools import partial
from typing import Any, Iterable, List, Union

from lightning import LightningModule
from torch.utils.data import DataLoader

from rl4co.data.generate_data import generate_default_datasets
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.optim_helpers import create_optimizer, create_scheduler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


import pickle
def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / (r_den + 1e-5)
    return r_val

def log_gradients_norm(parameters):
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm

class mySupervisedLearning(RL4COLitModule):

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        critic: CriticNetwork = None,
        critic_kwargs: dict = {},
        clip_range: float = 0.2,  # epsilon of PPO
        ppo_epochs: int = 2,  # inner epoch, K
        mini_batch_size: Union[int, float] = 0.25,  # 0.25,
        vf_lambda: float = 0.5,  # lambda of Value function fitting
        entropy_lambda: float = 0.0,  # lambda of entropy bonus
        normalize_adv: bool = False,  # whether to normalize advantage
        max_grad_norm: float = 0.5,  # max gradient norm
        metrics: dict = {
            "train": ["loss", "true_reward", "surrogate_loss", "value_loss", "entropy",
                "explained_variance", "gradient_norm", "adv_mean", "adv_std"],
        },
        quantile: float = 0.7,
        print_grads: bool = False,
        **kwargs,
    ):
        super().__init__(env, policy, metrics=metrics, **kwargs)
        self.automatic_optimization = False  # PPO uses custom optimization routine

        if critic is None:
            log.info("Creating critic network for {}".format(env.name))
            critic = create_critic_from_actor(policy, **critic_kwargs)
        self.critic = critic

        if isinstance(mini_batch_size, float) and (
            mini_batch_size <= 0 or mini_batch_size > 1
        ):
            default_mini_batch_fraction = 0.25
            log.warning(
                f"mini_batch_size must be an integer or a float in the range (0, 1], got {mini_batch_size}. Setting mini_batch_size to {default_mini_batch_fraction}."
            )
            mini_batch_size = default_mini_batch_fraction

        if isinstance(mini_batch_size, int) and (mini_batch_size <= 0):
            default_mini_batch_size = 128
            log.warning(
                f"mini_batch_size must be an integer or a float in the range (0, 1], got {mini_batch_size}. Setting mini_batch_size to {default_mini_batch_size}."
            )
            mini_batch_size = default_mini_batch_size

        self.ppo_cfg = {
            "clip_range": clip_range,
            "ppo_epochs": ppo_epochs,
            "mini_batch_size": mini_batch_size,
            "vf_lambda": vf_lambda,
            "entropy_lambda": entropy_lambda,
            "normalize_adv": normalize_adv,
            "max_grad_norm": max_grad_norm,
        }

        self.quantile = quantile
        self.print_grads = print_grads

        self.opt_policy = torch.optim.Adam(policy.parameters(), lr=1e-4, weight_decay=1e-6)
        self.opt_critic = torch.optim.Adam(critic.parameters(), lr=1e-4, weight_decay=1e-6)

    def configure_optimizers(self):
        parameters = list(self.policy.parameters()) + list(self.critic.parameters())
        return super().configure_optimizers(parameters)

    def on_train_epoch_end(self):
        """
        ToDo: Add support for other schedulers.
        """

        sch = self.lr_schedulers()

        # If the selected scheduler is a MultiStepLR scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.MultiStepLR):
            sch.step()

    def setup(self, stage="fit"):
        """Base LightningModule setup method. This will setup the datasets and dataloaders

        Note:
            We also send to the loggers all hyperparams that are not `nn.Module` (i.e. the policy).
            Apparently PyTorch Lightning does not do this by default.
        """

        print("#############Setup In MySupervisedLearning#############")

        log.info("Setting up batch sizes for train/val/test")
        train_bs, val_bs, test_bs = (
            self.data_cfg["batch_size"],
            self.data_cfg["val_batch_size"],
            self.data_cfg["test_batch_size"],
        )
        self.train_batch_size = train_bs
        self.val_batch_size = train_bs if val_bs is None else val_bs
        self.test_batch_size = self.val_batch_size if test_bs is None else test_bs

        if self.data_cfg["generate_default_data"]:
            log.info(
                "Generating default datasets. If found, they will not be overwritten"
            )
            generate_default_datasets(data_dir=self.data_cfg["data_dir"])

        log.info("Setting up datasets")
        self.train_dataset = self.wrap_dataset(
            self.env.dataset(self.data_cfg["train_data_size"], phase="train", filename="./tensordict/train/td_test.npz")
        )
        self.val_dataset = self.env.dataset(self.data_cfg["val_data_size"], phase="val")
        self.test_dataset = self.env.dataset(
            self.data_cfg["test_data_size"], phase="test"
        )
        self.dataloader_names = None
        self.setup_loggers()
        self.post_setup_hook()

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        # Evaluate old actions, log probabilities, and rewards
        if phase != "train":
            with torch.no_grad():
                td = self.env.reset(batch)  # note: clone needed for dataloader
                # print(batch)
                # print(td)
                # print(td["action_mask"])
                # print(td["actions"])
                # exit()
                # return
                batch = self.policy(td.clone(), self.env, phase=phase, return_actions=True)

                return batch


        if phase == "train":
            batch_size = batch["Graph"].shape[0]
            td = batch
            td.set("logprobs", batch["log_likelihood"])
            td.set("done", torch.zeros_like(td["done"]))
            # 这个batch就是一个TensorDict，和生成的数据集是相同的

            # infer batch size
            if isinstance(self.ppo_cfg["mini_batch_size"], float):
                mini_batch_size = int(batch_size * self.ppo_cfg["mini_batch_size"])
            elif isinstance(self.ppo_cfg["mini_batch_size"], int):
                mini_batch_size = self.ppo_cfg["mini_batch_size"]
            else:
                raise ValueError("mini_batch_size must be an integer or a float.")

            if mini_batch_size > batch_size:
                mini_batch_size = batch_size

            # Todo: Add support for multi dimensional batches
            # td.set("logprobs", out["log_likelihood"])
            # td.set("reward", out["reward"])
            # td.set("action", out["actions"])

            # Inherit the dataset class from the environment for efficiency
            dataset = self.env.dataset_cls(td)
            dataloader = DataLoader(
                dataset,
                batch_size=mini_batch_size,
                shuffle=True,
                collate_fn=dataset.collate_fn,
            )

            for _ in range(self.ppo_cfg["ppo_epochs"]):  # PPO inner epoch, K
                for sub_td in dataloader:
                    sub_td = sub_td.to(td.device)
                    previous_reward = sub_td["reward"].view(-1, 1)
                    out = self.policy(  # note: remember to clone to avoid in-place replacements!
                        sub_td.clone(),
                        actions=sub_td["actions"],
                        env=self.env,
                        return_entropy=True,
                        return_sum_log_likelihood=False,
                    )
                    ll, entropy = out["log_likelihood"], out["entropy"]
                    norm_reward = out["reward"]
                    true_reward = self.env.convert_to_true_reward(norm_reward)

                    # Compute the ratio of probabilities of new and old actions
                    # ratio = torch.exp(ll.sum(dim=-1) - sub_td["logprobs"]).view(
                    #     -1, 1
                    # )  # [batch, 1]

                    # Compute the advantage
                    value_pred = self.critic(sub_td)  # [batch, 1]
                    adv = previous_reward - value_pred.detach()

                    adv_mean, adv_std = torch.mean(adv), torch.std(adv)



                    explained_variance = 1 - (torch.var(previous_reward - value_pred) / torch.var(previous_reward))

                    # Normalize advantage
                    if self.ppo_cfg["normalize_adv"]:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Compute the surrogate loss
                    # surrogate_loss = -torch.min(
                    #     ratio * adv,
                    #     torch.clamp(
                    #         ratio,
                    #         1 - self.ppo_cfg["clip_range"],
                    #         1 + self.ppo_cfg["clip_range"],
                    #     )
                    #     * adv,
                    # ).mean()

                    negative_likelihood = -ll.sum(dim=-1).view(-1, 1)

                    # compute value function loss
                    #value_loss = F.huber_loss(value_pred, previous_reward)

                    # IQL's value function loss
                    vf_err = value_pred - previous_reward
                    vf_sign = (vf_err > 0).float()
                    quantile = self.quantile
                    vf_weight = (1 - vf_sign) * quantile + vf_sign * (1 - quantile)
                    value_loss = (vf_weight * (vf_err ** 2)).mean()

                    # compute total loss
                    # loss = (
                    #     surrogate_loss
                    #     + self.ppo_cfg["vf_lambda"] * value_loss
                    #     - self.ppo_cfg["entropy_lambda"] * entropy.mean()
                    # )
                    loss = (
                        negative_likelihood.mean()
                        - self.ppo_cfg["entropy_lambda"] * entropy.mean()
                    )

                    # breakpoint()

                    # td.set["loss"](loss)
                    self.opt_policy.zero_grad()
                    self.opt_critic.zero_grad()
                    self.backward(loss)
                    self.backward(value_loss)

                    # total_norm = log_gradients_norm(self.parameters())
                    # if self.print_grads:
                    #     print("_______________________________")
                    #     for name, param in self.named_parameters():
                    #         if param.grad is not None and torch.norm(param.grad) > 50:
                    #             print(f"{name}: {torch.norm(param.grad)}")


                    if self.ppo_cfg["max_grad_norm"] is not None:
                        self.clip_gradients(
                            self.opt_policy,
                            gradient_clip_val=self.ppo_cfg["max_grad_norm"],
                            gradient_clip_algorithm="norm",
                        )

                    self.opt_policy.step()
                    self.opt_critic.step()


            out.update(
                {
                    "loss": loss,
                    "BC_loss": negative_likelihood.mean(),
                    "value_loss": value_loss.mean(),
                    "entropy": entropy.mean(),
                    "explained_variance": explained_variance.mean(),
                    "true_reward": true_reward.mean(),
                    "adv_mean": adv_mean,
                    "adv_std": adv_std,
                }
            )

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
