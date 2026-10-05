# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

import os
from isaaclab.utils import configclass

@configclass
class RslRlMLPModelCfg:
    class_name: str = "MLPModel"
    hidden_dims: list[int] = MISSING
    activation: str = MISSING
    obs_normalization: bool = MISSING
    stochastic: bool = MISSING
    init_noise_std: float = 1.0
    noise_std_type: Literal["scalar", "log"] = "log"
    state_dependent_std: bool = False

@configclass
class RslRlMoEModelCfg:
    class_name: str = "MoEModel"
    num_experts: int = 16
    stochastic: bool = False
    init_noise_std: float = 1.0
    noise_std_type: str = "log"

@configclass
class RslRlAME2ModelCfg:
    class_name: str = "AME2Model"
    num_heads: int = 32
    stochastic: bool = True
    init_noise_std: float = 1.0
    noise_std_type: str = "log"

@configclass
class RslRlAME2LSIOModelCfg:
    class_name: str = "AME2LSIOModel"
    num_heads: int = 32
    stochastic: bool = True
    init_noise_std: float = 1.0
    noise_std_type: str = "log"

@configclass
class RslRlAME2GazeLSIOModelCfg:
    """Gaze student: the AME2GazeModel map path (global context -> gaze crop ->
    dense MHA) on the LSIO student inputs (neural_map + prop history + cmds).
    Same knobs as :class:`RslRlAME2GazeModelCfg`; the L_roi boundary penalty is
    picked up by PPO_IL via the model's auxiliary_loss() hook."""
    class_name: str = "rsl_rl.models.AME2_models:AME2GazeLSIOModel"
    num_heads: int = 32
    crop_x: int = 12  # gaze crop size in cells along map x (forward)
    crop_y: int = 10  # gaze crop size in cells along map y (lateral)
    roi_coef: float = 0.01
    roi_margin: float = 0.45
    stochastic: bool = True
    init_noise_std: float = 1.0
    noise_std_type: str = "log"

@configclass
class RslRlAME2GazeModelCfg:
    """TAGA-style active gaze (arXiv:2606.05880) + dense-attention crop encoder.

    Straight-through crop keeps original height values (bilinear gradients only);
    the gaze boundary penalty (roi) is added to the PPO loss via the model's
    auxiliary_loss() hook and logged as Loss/auxiliary.
    """
    class_name: str = "AME2GazeModel"
    num_heads: int = 32
    crop_x: int = 12  # gaze crop size in cells along map x (forward)
    crop_y: int = 10  # gaze crop size in cells along map y (lateral)
    roi_coef: float = 0.01  # TAGA L_roi weight; small suffices per the authors
    roi_margin: float = 0.45  # penalty-free zone |r - 0.5| <= margin
    stochastic: bool = True
    init_noise_std: float = 1.0
    noise_std_type: str = "log"

@configclass
class RslRlPpoAlgorithmCfg:
    class_name: str = "PPO"
    # "muon": hybrid Muon (ndim>=2 weights) + AdamW (biases/1D), RMS-matched to reuse the Adam learning rate.
    # Checkpoints trained with "adam" cannot resume with "muon" optimizer state.
    optimizer: str = "muon"
    # bfloat16 autocast for the update's forward pass and loss only (adapted from rsl_rl PR #219);
    # weights, backward, optimizer and exports stay fp32. Most beneficial on Blackwell GPUs.
    use_mixed_precision: bool = True
    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    learning_rate: float = MISSING
    schedule: str = MISSING
    gamma: float = MISSING
    lam: float = MISSING
    entropy_coef: float = MISSING
    desired_kl: float = MISSING
    max_grad_norm: float = MISSING
    value_loss_coef: float = MISSING
    use_clipped_value_loss: bool = MISSING
    clip_param: float = MISSING
    normalize_advantage_per_mini_batch: bool = False
    ent_coef_decay: float = 0.9999
    min_ent_coef: float = 0.001
    renorm_max_value_loss: float = 100.0
    rnd_cfg: dict | None = None
    symmetry_cfg: dict | None = None

@configclass
class RslRlPpoIlAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "rsl_rl.algorithms.ppo_il:PPO_IL"
    teacher_path: str = MISSING
    imitation_coef: float = 0.02
    repr_coef: float = 0.2
    rl_switch_point: int = 4001
    switched_lr: float = 1e-5
    initial_lr: float = 0.001

@configclass
class RslRlBaseRunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    empirical_normalization: bool | None = None
    obs_groups: dict[str, list[str]] = MISSING
    clip_actions: float | None = None
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "isaaclab"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    torch_compile_mode: str | None = None

@configclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "OnPolicyRunner"
    actor: RslRlMLPModelCfg | RslRlMoEModelCfg | RslRlAME2ModelCfg | RslRlAME2LSIOModelCfg | RslRlAME2GazeModelCfg = MISSING
    critic: RslRlMLPModelCfg | RslRlMoEModelCfg = MISSING
    algorithm: RslRlPpoAlgorithmCfg = MISSING


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 2000
    experiment_name = "ame2"
    obs_groups = {"actor": ["teacher_mapping", "teacher_prop"], "critic": ["critic"]}
    actor = RslRlAME2ModelCfg(
        num_heads=32,
        stochastic=True,
        init_noise_std=1.0,
        noise_std_type="log",
    )
    critic = RslRlMoEModelCfg(
        num_experts=16,
        stochastic=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        ent_coef_decay=0.9999,
        min_ent_coef=0.001,
        renorm_max_value_loss=100.0,
        symmetry_cfg={
            "use_data_augmentation": True,
            "use_mirror_loss": False,
            "data_augmentation_func": "ame2.tasks.mdp.symmetry:ame2_g1_symmetry_augmentation",
            "for_critic_only": True,
        },
    )


@configclass
class PPORunnerCfg_G1_Gaze(PPORunnerCfg):
    """G1 teacher with the TAGA-gaze actor; same env/algorithm as PPORunnerCfg."""

    experiment_name = "ame2_gaze"
    actor = RslRlAME2GazeModelCfg()


@configclass
class PPORunnerCfg_G1_Gaze_Student(PPORunnerCfg):
    """LSIO student for the G1 gaze teacher.

    Consumes the three student observation groups (``neural_map`` from the
    Mid360 livox pipeline, ``student_prop_hist``, ``student_cmds``) and distills
    from a frozen gaze-teacher JIT. The ``_TorchAME2GazeModel`` export returns
    (actions, crop-attention latent, global context) — exactly the 3 tensors
    ``PPO_IL`` expects. Drop the exported teacher at the path below (or override
    ``algorithm.teacher_path``) before training.
    """

    experiment_name = "ame2_g1_gaze_stud"
    obs_groups = {"actor": ["neural_map", "student_prop_hist", "student_cmds"], "critic": ["critic"]}
    # same gaze crop as the teacher; use RslRlAME2LSIOModelCfg for the dense-attention LSIO baseline
    actor = RslRlAME2GazeLSIOModelCfg()
    resume = False
    max_iterations = 20000
    algorithm = RslRlPpoIlAlgorithmCfg(
        teacher_path=os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..",
            "NN_models", "g1_gaze_teacher_isaaclab.jit",
        )),
        imitation_coef=0.02,
        repr_coef=0.2,
        rl_switch_point=4001,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        ent_coef_decay=0.9999,
        min_ent_coef=0.0005,
        renorm_max_value_loss=20.0,
        symmetry_cfg={
            "use_data_augmentation": True,
            "use_mirror_loss": False,
            "data_augmentation_func": "ame2.tasks.mdp.symmetry:ame2_g1_symmetry_augmentation",
            "for_critic_only": True,
        },
    )
    load_run = ".*"
