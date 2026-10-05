# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class PPO_IL(PPO):
    """PPO combined with teacher-student imitation learning (distillation).

    Symmetry data augmentation is supported for the critic only.
    """

    # Inherits the multi-head critic (default on): update() handles the value loss and combined
    # advantage through the same dimension-agnostic code as base PPO, and the student runs under
    # OnPolicyRunner (which injects the per-component rewards). The multi-head critic helps the RL
    # fine-tuning phase (after ``rl_switch_point``); during pure distillation the critic is still
    # trained but the actor does not use its advantages.

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        teacher_path: str,
        imitation_coef: float = 0.02,
        repr_coef: float = 0.2,
        rl_switch_point: int = 5001,
        initial_lr: float = 0.001,
        switched_lr: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        
        # Teacher model
        self.teacher_path = teacher_path
        self.teacher = torch.jit.load(self.teacher_path, map_location=self.device)
        self.teacher.eval() # Teacher is always in eval mode

        # Distillation parameters
        self.imitation_coef = imitation_coef
        self.repr_coef = repr_coef
        self.rl_switch_point = rl_switch_point
        self.update_cnt = 0
        self.switched_lr = switched_lr
        self.initial_lr = initial_lr

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        self.update_cnt += 1
        
        # Handle switch point for Learning Rate
        if self.update_cnt == self.rl_switch_point:
            print(f"[PPO_IL] Reached switch point ({self.rl_switch_point}). Dropping LR to {self.switched_lr}")
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.switched_lr
            self.learning_rate = self.switched_lr

        # coefficients
        ts_coef = 1.0 # Distillation is always on
        rl_coef = 1.0 if self.update_cnt >= self.rl_switch_point else 0.0

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_imit_loss = 0
        mean_repr_loss = 0
        # Model-defined auxiliary loss (e.g. the gaze models' boundary penalty);
        # mirrors base PPO — valid right after the student forward in step 3.
        mean_aux_loss = 0 if hasattr(self._raw_actor, "auxiliary_loss") else None

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]
            num_aug = 1
            
            # --- 0. Advantage Normalization ---
            if self.normalize_advantage_per_mini_batch:
                batch.advantages = self._normalize_advantages(batch.advantages)

            # --- 1. Critic-Only Symmetry Augmentation ---
            critic_obs_final = batch.observations
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Only critic-side augmentation is supported
                assert self.symmetry.get("for_critic_only", False), "PPO_IL currently only supports symmetry augmentation for critic only."
                data_augmentation_func = self.symmetry["data_augmentation_func"]

                # Augment only critic observations
                critic_obs_final, _ = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=None,
                    for_critic=True,
                )
                num_aug = int(critic_obs_final.batch_size[0] / original_batch_size)
                
                # Expand targets for critic
                batch_values = batch.values.repeat(num_aug, 1)
                batch_returns = batch.returns.repeat(num_aug, 1)
                batch_masks_C = batch.masks.repeat(num_aug, 1) if batch.masks is not None else None
                batch_h1_S = batch.hidden_states[1].repeat(num_aug, 1) if batch.hidden_states[1] is not None else None
            else:
                batch_values = batch.values
                batch_returns = batch.returns
                batch_masks_C = batch.masks
                batch_h1_S = batch.hidden_states[1]

            # --- 2. Teacher Inference (Original Samples) ---
            with torch.no_grad():
                teacher_output = self._teacher_forward(batch)
                action_T, local_T, global_T = teacher_output[:3]

            # Optionally run the student forward pass and loss computation in bfloat16
            # autocast (adapted from PR leggedrobotics/rsl_rl#219; flag inherited from PPO).
            # The frozen fp32 JIT teacher above stays outside; backward/clip/step stay fp32.
            with torch.amp.autocast(
                device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
            ):
                # --- 3. Student Inference (distillation pass) ---
                local_S, global_S = self._student_forward(batch, teacher_output)
                # Log prob of the teacher actions under the student policy (distillation pass)
                actions_log_prob_teacher = self._imitation_log_prob(action_T, teacher_output)
                # --- Distillation losses (read the distillation pass only) ---
                imit_loss = -1. * actions_log_prob_teacher.mean()
                repr_loss = F.mse_loss(local_S, local_T) + F.mse_loss(global_S, global_T)
                distill_loss = ts_coef * (self.imitation_coef * imit_loss + self.repr_coef * repr_loss)

                # --- 4. Adaptive KL Schedule ---
                distribution_params = self.actor.output_distribution_params
                if self.update_cnt >= self.rl_switch_point and self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                        kl_mean = torch.mean(kl)

                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                            kl_mean /= self.gpu_world_size

                        if self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-3, self.learning_rate * 1.5)

                        if self.is_multi_gpu:
                            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                            torch.distributed.broadcast(lr_tensor, src=0)
                            self.learning_rate = lr_tensor.item()

                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                # --- 5. Ordinary PPO logic ---
                # Distribution is already set up by forward() with stochastic_output=True
                # Log prob of the stored batch actions
                actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            
                # Surrogate (skipped before the switch: an overflowing ratio would give 0 * inf = NaN grads)
                if rl_coef > 0.0:
                    # log-ratio clamp: inactive in healthy training (|log r| << 20), prevents exp overflow
                    ratio = torch.exp(torch.clamp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob), -20.0, 20.0))
                    surrogate = -torch.squeeze(batch.advantages) * ratio
                    surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                else:
                    surrogate_loss = torch.zeros((), device=self.device)

                # Value (Augmented if enabled)
                values = self.critic(critic_obs_final, masks=batch_masks_C, hidden_state=batch_h1_S)
                if self.use_clipped_value_loss:
                    value_clipped = batch_values + (values - batch_values).clamp(-self.clip_param, self.clip_param)
                    v_loss = (values - batch_returns).pow(2)
                    v_loss_clipped = (value_clipped - batch_returns).pow(2)
                    value_loss = torch.max(v_loss, v_loss_clipped).mean()
                else:
                    value_loss = (batch_returns - values).pow(2).mean()

                if self.renorm_max_value_loss is not None and value_loss.item() > self.renorm_max_value_loss:
                    value_loss = value_loss * (self.renorm_max_value_loss / value_loss.detach())

                entropy = self.actor.output_entropy.mean()

                # --- 7. Final Combined Loss ---
                total_loss = (
                    rl_coef * (surrogate_loss - self.entropy_coef * entropy) +
                    self.value_loss_coef * value_loss
                ) + distill_loss
                # Model-defined auxiliary loss (weighted inside the model; reads the
                # student forward that ran in step 3)
                if mean_aux_loss is not None:
                    aux_loss = self._raw_actor.auxiliary_loss()
                    total_loss = total_loss + aux_loss

            # Backprop
            self.optimizer.zero_grad()
            total_loss.backward()
            # Average gradients across GPUs (as PPO.update does); without it the ranks drift apart
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Logging
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.item()
            mean_imit_loss += imit_loss.item()
            mean_repr_loss += repr_loss.item()
            if mean_aux_loss is not None:
                mean_aux_loss += aux_loss.item()

        # Finalize Logging
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_imit_loss /= num_updates
        mean_repr_loss /= num_updates
        if mean_aux_loss is not None:
            mean_aux_loss /= num_updates

        self.storage.clear()

        # Decay entropy coefficient
        if self.entropy_coef > self.min_ent_coef:
            self.entropy_coef *= self.ent_coef_decay
            self.entropy_coef = max(self.entropy_coef, self.min_ent_coef)

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "entropy_coef": self.entropy_coef,
            "imitation": mean_imit_loss,
            "representation": mean_repr_loss,
            "rl_coef": rl_coef,
            "update_cnt": self.update_cnt,
        }
        if mean_aux_loss is not None:
            loss_dict["auxiliary"] = mean_aux_loss
        return loss_dict

    # --- distillation-pass hooks ---

    def _teacher_forward(self, batch) -> tuple[torch.Tensor, ...]:
        """Frozen JIT teacher on the privileged obs -> (action_T, local_T, global_T, ...)."""
        prop_T = batch.observations["teacher_prop"].to(self.device)
        map_T = batch.observations["teacher_mapping"].to(self.device)
        teacher_output = self.teacher(prop_T, map_T)
        assert len(teacher_output) == 3, f"Teacher JIT returns 3 tensors, got {len(teacher_output)}"
        return teacher_output

    def _student_forward(self, batch, teacher_output) -> tuple[torch.Tensor, torch.Tensor]:
        """Student forward for the distillation losses; returns (local_S, global_S)."""
        _dummy_action_S_sampled, local_S, global_S = self.actor(
            batch.observations,
            masks=batch.masks,
            hidden_state=batch.hidden_states[0],
            return_latent=True,
            stochastic_output=True,
        )
        return local_S, global_S

    def _imitation_log_prob(self, action_T: torch.Tensor, teacher_output) -> torch.Tensor:
        """Log prob of the teacher's actions under the student's distillation-pass distribution."""
        return self.actor.get_output_log_prob(action_T)


    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO_IL:
        """Construct the PPO_IL algorithm."""
        # Resolve class callables
        alg_class: type[PPO_IL] = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Symmetry configuration
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Extract IL config
        alg_cfg = cfg["algorithm"].copy()
        alg_cfg.pop("class_name", None)
        teacher_path = alg_cfg.pop("teacher_path", None)
        assert teacher_path is not None, "teacher_path must be provided for PPO_IL distillation."

        # Initialize the algorithm
        alg: PPO_IL = alg_class(
            actor, 
            critic, 
            storage, 
            teacher_path=teacher_path,
            device=device, 
            **alg_cfg,
            multi_gpu_cfg=cfg.get("multi_gpu")
        )

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode", None))

        return alg
