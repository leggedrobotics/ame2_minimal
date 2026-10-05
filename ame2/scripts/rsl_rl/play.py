# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--plot", action="store_true", default=False, help="Enable debug plotting of base velocity and joint data.")
parser.add_argument("--log_rewards", action="store_true", default=False, help="Log every reward term over a single episode (env_idx=0) and plot the curves on termination.")
parser.add_argument("--export", action="store_true", default=False, help="Export AME2 policy to JIT optimize_for_inference model.")
parser.add_argument("--vizacc", action="store_true", default=False, help="Open a live matplotlib plot of hip-link |a_w| (env 0) during play.")
parser.add_argument("--fly", action="store_true", default=False, help="Apply an external world-frame force on the robot base that lifts and flies it toward the current goal.")
parser.add_argument("--viz_lidar", action="store_true", default=False, help="Debug: hide all other overlays and draw ONLY the lidar_cam hit points plus lines from the sensor origin to each hit (env 0).")
parser.add_argument("--return_home", action="store_true", default=False, help="Once a robot gets within 0.5 m of its commanded goal, re-target the command to its episode spawn pose so it walks there and back (exercises the neural-map memory on the return leg).")
parser.add_argument("--no_markers", action="store_true", default=False, help="Disable ALL visualization overlays (height-scan/attention/neural-map markers, sensor debug dots, goal arrows) for clean renders, e.g. video recording.")
parser.add_argument("--terrain", type=str, default=None, help="Select a single sub-terrain by name from the FULL training generator (overrides the play cfg's hardcoded pick), e.g. climb_up, stones, gap.")
parser.add_argument("--difficulty", type=float, default=None, help="Pin the terrain difficulty (sets difficulty_range=(d, d)).")
parser.add_argument("--terrain_height", type=float, default=None, help="Pin the selected terrain's height (sets min_height = max_height, making it difficulty-independent); climb_up / climb_down style terrains only.")
parser.add_argument("--goal_x_range", type=float, nargs=2, default=None, help="Override the selected terrain's goal-patch x sampling range (tile-local, m) so commanded goals land far across the terrain feature; requires --terrain.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply, wrap_to_pi, normalize
from isaaclab.utils.dict import print_dict
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import ame2.tasks  # noqa: F401


def _plot_reward_curves(reward_log: dict, term_names: list[str], log_dir: str, dt: float) -> None:
    """Plot per-step reward-term curves for one episode and save as PDF.

    ``reward_log[name]`` holds the per-step weighted contribution in units of
    *reward / second* (matches ``RewardManager._step_reward``). The cumulative
    sum is therefore obtained by integrating with ``dt``.
    """
    t = np.array(reward_log["time"])
    total = np.array(reward_log["total"])
    n = len(term_names)
    num_cols = 4
    num_rows = math.ceil(n / num_cols) + 1  # +1 row for the "total" subplot

    fig = plt.figure(figsize=(4 * num_cols, 3 * num_rows))

    # Top: total reward — instantaneous (left axis) and cumulative integral (right axis).
    ax_top = plt.subplot2grid((num_rows, num_cols), (0, 0), colspan=num_cols)
    ax_top.plot(t, total, label="Total (per second)", color="C0")
    ax_top.axhline(0.0, color="gray", linewidth=0.5)
    ax_top.set_xlabel("Time [s]")
    ax_top.set_ylabel("reward / s")
    ax_top.grid(True)
    ax_top_r = ax_top.twinx()
    ax_top_r.plot(t, np.cumsum(total) * dt, color="C1", linestyle="--", label="Cumulative")
    ax_top_r.set_ylabel("∫ reward dt")
    ax_top.set_title(f"Total reward (cumulative ∑ = {float(np.sum(total) * dt):+.3f})")
    ax_top.legend(loc="upper left")
    ax_top_r.legend(loc="upper right")

    # Per-term subplots — sort by |cumulative| descending so the most influential terms come first.
    cumulative = {name: float(np.sum(reward_log[name]) * dt) for name in term_names}
    sorted_names = sorted(term_names, key=lambda k: abs(cumulative[k]), reverse=True)

    for i, name in enumerate(sorted_names):
        r, c = (i // num_cols) + 1, i % num_cols
        ax = plt.subplot2grid((num_rows, num_cols), (r, c))
        vals = np.array(reward_log[name])
        ax.plot(t, vals, color="C0")
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.set_title(f"{name}\n∑ = {cumulative[name]:+.3f}")
        ax.set_xlabel("Time [s]")
        ax.grid(True)

    plt.tight_layout()
    out = os.path.join(log_dir, "reward_curves.pdf")
    plt.savefig(out)
    plt.close(fig)
    print(f"[INFO] LOG_REWARDS: saved per-term reward curves to {out}")
    print("[INFO] LOG_REWARDS: top contributors by |∑| over the episode:")
    for name in sorted_names[: min(8, len(sorted_names))]:
        print(f"    {name:40s}  ∑ = {cumulative[name]:+.4f}")


def _build_hip_acc_plot(env, term_name: str = "early_hip_acc", window_seconds: float = 10.0,
                        warmup_skip: float = 0.1):
    """Spawn a sibling Python process that live-plots hip-link ``|a_w|`` via matplotlib.

    The child runs its own event loop on the WebAgg backend with a sanitized
    env, isolated from Isaac Sim's Qt setup. The parent streams ``{t, norms}``
    JSON lines on the child's stdin, one per sim step.

    Returns ``update(env_idx)`` to call each sim step, or ``None`` if the
    termination term / sensor / plotter script cannot be located.
    """
    try:
        term_cfg = env.unwrapped.termination_manager.get_term_cfg(term_name)
    except Exception as e:
        print(f"[WARN] Termination term '{term_name}' not found ({e}); skipping hip_acc plot.")
        return None

    asset_cfg = term_cfg.params["asset_cfg"]
    sensor_cfg = term_cfg.params["sensor_cfg"]
    threshold = float(term_cfg.params.get("threshold", 100.0))   # drawn as a reference line

    robot = env.unwrapped.scene[asset_cfg.name]
    accel_sensor = env.unwrapped.scene.sensors[sensor_cfg.name]
    # mirror the lazy attachment done by ame2_body_lin_acc_out_of_limit
    if getattr(accel_sensor, "_asset", None) is None:
        accel_sensor.set_asset(robot)

    body_ids = list(asset_cfg.body_ids)
    body_names = [robot.body_names[i] for i in body_ids]

    plotter_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_hip_acc_plotter.py")
    if not os.path.exists(plotter_path):
        print(f"[WARN] hip_acc plotter script missing: {plotter_path}")
        return None

    import json
    import subprocess

    # Sanitized child env:
    # PYTHONPATH: forward sys.path (Isaac Sim modifies it at runtime) so matplotlib/tornado import.
    # MPLBACKEND=WebAgg: auto-selected Qt fails against the incompatible Qt on Isaac Sim's LD_LIBRARY_PATH.
    # QT_*: removed as a precaution.
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    for var in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        child_env.pop(var, None)
    child_env["MPLBACKEND"] = "WebAgg"
    backend_hint = "WebAgg (the child will print the http://… URL — open it in any browser)"

    try:
        proc = subprocess.Popen(
            [
                sys.executable, "-u", plotter_path,
                "--body_names", ",".join(body_names),
                "--threshold", str(threshold),
                "--window", str(window_seconds),
            ],
            stdin=subprocess.PIPE,
            env=child_env,
        )
    except Exception as e:
        print(f"[WARN] Failed to spawn hip_acc plot subprocess: {e}")
        return None
    print(
        f"[INFO] hip_acc plot subprocess: pid={proc.pid} backend_hint={backend_hint} "
        f"bodies={body_names} threshold={threshold:.1f} m/s² "
        f"window={window_seconds:.1f}s warmup_skip={warmup_skip:.2f}s"
    )

    def update(env_idx: int = 0) -> None:
        if proc.poll() is not None:
            return  # plot window was closed / child exited
        try:
            # skip the settling transient at episode start (independent of the termination's start_time)
            cur_t = float((env.unwrapped.episode_length_buf[env_idx] * env.unwrapped.step_dt).item())
            if cur_t < warmup_skip:
                return
            acc = accel_sensor.data.acc_w[env_idx][body_ids]                # (B, 3)
            acc_norm = acc.norm(dim=-1).detach().cpu().numpy().tolist()      # (B,)
            payload = (json.dumps({"t": cur_t, "norms": acc_norm}) + "\n").encode()
            try:
                proc.stdin.write(payload)
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass  # plotter exited; future calls short-circuit via proc.poll()
        except Exception as e:
            print(f"[WARN] hip_acc plot update error: {e}")

    return update


class FlyController:
    """Flies the robot base to its goal via an external world-frame wrench on the base body.

    Two decoupled PD loops, both in the world frame and applied together with ``is_global=True``:

    * **Position** — pulls the base toward the goal (held at least ``clearance`` above the tallest
      terrain the height scanner sees) and compensates gravity, so the robot lifts and clears
      obstacles. ``force = mass * accel``.
    * **Attitude** — a *direct* torque (in N·m, clamped to ``max_torque``) that levels the body and
      yaws it to face the goal. Torque is not scaled by the link inertia, since the per-link root
      inertia reported by the sim is an unreliable placeholder (~0.01 kg·m²).

    The wrench is applied to the *heaviest* link (the trunk), not body 0: body 0 is a nearly
    massless root frame, and a torque there spins it up uncontrollably. All control state
    (pose, velocity, heading) is read from that same link.
    """

    def __init__(self, env, command_name: str = "base_pose"):
        self.robot = env.unwrapped.scene["robot"]
        self.command = env.unwrapped.command_manager.get_term(command_name)
        self.scanner = env.unwrapped.scene["height_scanner"]
        device = env.unwrapped.device
        # apply force/torque to the heaviest link (real trunk), not the massless root frame
        masses = self.robot.root_physx_view.get_masses().to(device)         # (E, num_bodies)
        self.body_idx = int(masses[0].argmax().item())
        self.body_ids = [self.body_idx]
        self.mass = masses.sum(dim=1)                                       # (E,) total robot mass
        self.up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(env.unwrapped.num_envs, 3)
        self.forward = torch.tensor([1.0, 0.0, 0.0], device=device).expand(env.unwrapped.num_envs, 3)

        # position loop (force = mass * accel)
        self.kp_pos = torch.tensor([1.0, 1.0, 10.0], device=device)  # x, y, z gains
        self.kd_pos = 10.0
        self.max_acc = 2.0    # [m/s^2] clamp
        self.gravity = 9.81
        self.clearance = 0.8    # [m] min height above the tallest sensed terrain point
        self.fly_height_radius = 0.6  # [m] only terrain within this horizontal radius of the
                                      # base sets the lift height (small local scanner, not far-ahead obstacles)

        # attitude loop (direct torque, N·m)
        self.kp_ang = 20.0     # roll/pitch leveling [N·m/rad]
        self.kd_ang = 5.0      # roll/pitch damping  [N·m/(rad/s)]
        self.kp_yaw = 10.0      # yaw-to-goal        [N·m/rad]
        self.kd_yaw = 3.0      # yaw damping         [N·m/(rad/s)]
        self.max_torque = 40.0     # [N·m] per-axis clamp
        self.yaw_deadzone = 0.3     # [m] no yaw-to-goal within this horizontal distance to goal
        self.yaw_level_tol = 0.35   # [rad] only yaw-correct when tilt < this (~20 deg)

        # recompute/reapply the wrench only every N control steps; in between, the sim keeps
        # applying the last set wrench (the external-wrench buffer persists), so the effective
        # update rate is lower / smoother.
        self.update_interval = 5
        self._step_count = 0

        body_name = self.robot.body_names[self.body_idx]
        print(
            f"[INFO] --fly enabled: flying base to goal '{command_name}', wrench on heaviest link "
            f"'{body_name}' (idx {self.body_idx}, {float(masses[0, self.body_idx]):.1f} kg) "
            f"(max_acc={self.max_acc} m/s^2, clearance={self.clearance} m, "
            f"max_torque={self.max_torque} N·m, mean_mass={float(self.mass.mean()):.1f} kg)."
        )

    def _lifted_goal(self) -> torch.Tensor:
        """Goal position with z raised to keep the base above the tallest local terrain.

        Only height-scan hits within ``fly_height_radius`` of the robot base count, so
        the lift tracks the ground directly under/around the robot rather than far-ahead
        obstacles still inside the (forward-offset) scan grid.
        """
        goal = self.command.pos_command_w.clone()                          # (E, 3)
        hits = self.scanner.data.ray_hits_w                                 # (E, R, 3)
        hits_z = hits[..., 2]
        base_xy = self.robot.data.root_pos_w[:, :2]                        # (E, 2) base/pelvis
        horiz_d = torch.norm(hits[..., :2] - base_xy.unsqueeze(1), dim=-1)  # (E, R)
        valid = torch.isfinite(hits_z) & (horiz_d <= self.fly_height_radius)
        hits_z = torch.where(valid, hits_z, torch.full_like(hits_z, float("-inf")))
        terrain_z = hits_z.max(dim=1).values                               # (E,) -inf if nothing sensed nearby
        seen = torch.isfinite(terrain_z)
        # height from local terrain only (terrain_z + clearance); goal z only if nothing is sensed nearby
        goal[:, 2] = torch.where(seen, terrain_z + self.clearance, goal[:, 2])
        return goal

    def step(self) -> None:
        """Compute and apply the lift force + attitude torque, once every ``update_interval`` steps.

        On the skipped steps the simulation keeps applying the previously set wrench.
        """
        recompute = (self._step_count % self.update_interval) == 0
        self._step_count += 1
        if not recompute:
            return

        goal = self._lifted_goal()
        # all control state comes from the heavy link the wrench acts on
        idx = self.body_idx
        pos = self.robot.data.body_pos_w[:, idx]
        quat = self.robot.data.body_quat_w[:, idx]
        lin_vel = self.robot.data.body_lin_vel_w[:, idx]
        ang_vel = self.robot.data.body_ang_vel_w[:, idx]

        # --- position: world-frame PD + gravity compensation ---
        acc = (self.kp_pos * (goal - pos) - self.kd_pos * lin_vel)
        acc = acc.clamp(-self.max_acc, self.max_acc)
        acc[:, 2] += self.gravity
        force = (self.mass.unsqueeze(-1) * acc).unsqueeze(1)               # (E, 1, 3)

        # --- attitude: world-frame torque PD (direct N·m) ---
        b_z = quat_apply(quat, self.up)                                    # body z-axis in world
        tilt_angle = torch.acos(b_z[:, 2].clamp(-1.0, 1.0))               # (E,) tilt of body-z from up
        # angle-proportional error (does not weaken past 90 deg, so the body always rights itself)
        tilt_err = normalize(torch.cross(b_z, self.up, dim=-1)) * tilt_angle.unsqueeze(-1)
        torque = self.kp_ang * tilt_err - self.kd_ang * ang_vel           # roll/pitch on world x, y
        # yaw toward the goal, but only when nearly level (else yaw torque would tumble a tilted body)
        to_goal = goal[:, :2] - pos[:, :2]
        forward_w = quat_apply(quat, self.forward)                         # body forward in world
        heading = torch.atan2(forward_w[:, 1], forward_w[:, 0])
        yaw_err = wrap_to_pi(torch.atan2(to_goal[:, 1], to_goal[:, 0]) - heading)
        gate = (torch.norm(to_goal, dim=-1) > self.yaw_deadzone) & (tilt_angle < self.yaw_level_tol)
        torque[:, 2] = self.kp_yaw * yaw_err * gate - self.kd_yaw * ang_vel[:, 2]
        torque = torque.clamp(-self.max_torque, self.max_torque).unsqueeze(1)  # (E, 1, 3)

        self.robot.set_external_force_and_torque(force, torque, body_ids=self.body_ids, is_global=True)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # clean render: also silence the env-side debug drawings (sensor ray dots,
    # goal-pose arrows); play.py's own markers are skipped in the play loop
    if args_cli.no_markers:
        if getattr(env_cfg.scene, "height_scanner", None) is not None:
            env_cfg.scene.height_scanner.debug_vis = False
        if getattr(env_cfg.commands, "base_pose", None) is not None:
            env_cfg.commands.base_pose.debug_vis = False

    # single-terrain selection: re-filter from the full training generator (the
    # play cfg's __post_init__ already narrowed sub_terrains to its hardcoded
    # pick, so the original module-level cfg is the source of truth here)
    if args_cli.terrain is not None:
        import copy as _copy
        from ame2.tasks.terrains import AME2_TERRAINS_CFG_G1 as _FULL_TG
        if args_cli.terrain not in _FULL_TG.sub_terrains:
            raise ValueError(
                f"Unknown terrain '{args_cli.terrain}'. Available: {sorted(_FULL_TG.sub_terrains)}")
        _sel = _copy.deepcopy(_FULL_TG.sub_terrains[args_cli.terrain])
        _sel.proportion = 1.0
        if args_cli.terrain_height is not None:
            if not hasattr(_sel, "min_height") or not hasattr(_sel, "max_height"):
                raise ValueError(f"--terrain_height: terrain '{args_cli.terrain}' has no min/max_height")
            # height = min + (max - min) * difficulty -> min == max pins it exactly
            _sel.min_height = args_cli.terrain_height
            _sel.max_height = args_cli.terrain_height
        if args_cli.goal_x_range is not None:
            _sel.flat_patch_sampling["target"].x_range = tuple(args_cli.goal_x_range)
        env_cfg.scene.terrain.terrain_generator.sub_terrains = {args_cli.terrain: _sel}
    if args_cli.difficulty is not None:
        env_cfg.scene.terrain.terrain_generator.difficulty_range = (args_cli.difficulty, args_cli.difficulty)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy_with_attention(device=env.unwrapped.device)

    # attention markers: shades from blue (low) to red (high)
    height_scanner_sensor = env.unwrapped.scene["height_scanner"]
    height_scanner_critic_sensor = env.unwrapped.scene["height_scanner_critic"]
    # disable default debug visualization to avoid overlap with attention markers
    height_scanner_sensor.set_debug_vis(False)
    height_scanner_critic_sensor.set_debug_vis(False)
    
    # define 11 prototypes for different attention levels
    num_shades = 11
    markers_dict = {}
    for i in range(num_shades):
        ratio = i / (num_shades - 1)
        # blue (0,0,1) to red (1,0,0)
        color = (ratio, 0.0, 1.0 - ratio)
        markers_dict[f"shade_{i}"] = sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )

    attention_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/AttentionWeights",
        markers=markers_dict,
    )
    attention_markers = VisualizationMarkers(attention_marker_cfg)

    # processed heights seen by the critic (yellow) and actor (green)
    critic_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CriticObs",
        markers={"processed": sim_utils.SphereCfg(radius=0.02, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)))},
    )
    critic_markers = VisualizationMarkers(critic_marker_cfg)
    
    actor_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ActorObs",
        markers={"processed": sim_utils.CuboidCfg(size=(0.015, 0.015, 0.015), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)))},
    )
    actor_markers = VisualizationMarkers(actor_marker_cfg)

    # student visualizations: only if a neural_map sensor is present
    is_student_task = "neural_map" in env.unwrapped.scene.sensors
    
    if is_student_task:
        # neural map observation markers (attention-colored)
        num_shades = 11
        nm_markers_dict = {}
        nm_unc_markers_dict = {}
        for i in range(num_shades):
            ratio = i / (num_shades - 1)
            # blue (0,0,1) to red (1,0,0)
            color = (ratio, 0.0, 1.0 - ratio)
            nm_markers_dict[f"shade_{i}"] = sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            nm_unc_markers_dict[f"shade_{i}"] = sim_utils.CylinderCfg(
                radius=0.005,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        
        neural_map_est_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/NeuralMapEst",
            markers=nm_markers_dict,
        )
        neural_map_est_markers = VisualizationMarkers(neural_map_est_cfg)
        
        neural_map_unc_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/NeuralMapUnc",
            markers=nm_unc_markers_dict,
        )
        neural_map_unc_markers = VisualizationMarkers(neural_map_unc_cfg)

        # local scan grid (per-frame mapping model input): orange
        local_scan_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/LocalScanGrid",
            markers={"scan": sim_utils.SphereCfg(radius=0.015, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)))},
        )
        local_scan_markers = VisualizationMarkers(local_scan_cfg)

        # raw lidar hits: green
        raw_pcl_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/RawCameraHits",
            markers={"hit": sim_utils.SphereCfg(radius=0.01, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1)))},
        )
        raw_pcl_markers = VisualizationMarkers(raw_pcl_cfg)
    else:
        neural_map_est_markers = None
        neural_map_unc_markers = None
        local_scan_markers = None
        raw_pcl_markers = None

    # debug-draw interface for --viz_lidar (available only after the sim app launched)
    if args_cli.viz_lidar:
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
        lidar_draw = omni_debug_draw.acquire_debug_draw_interface()
        if "lidar_cam" not in env.unwrapped.scene.sensors:
            print("[WARN] --viz_lidar set but scene has no 'lidar_cam' sensor; nothing to draw.")
    else:
        lidar_draw = None

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    if hasattr(runner.alg, "get_policy"):
        policy_nn = runner.alg.get_policy()
    else:
        try:
            # version 2.3 onwards
            policy_nn = runner.alg.policy
        except AttributeError:
            # version 2.2 and below
            policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    dt = env.unwrapped.step_dt

    # --- Fly mode: external wrench lifts the robot to its goal (see FlyController) ---
    fly_controller = FlyController(env) if args_cli.fly else None

    # --- Return-home mode: remember each env's spawn pose; once the robot reaches
    # its goal, re-target the pose command to the spawn. The command is not
    # resampled mid-episode (16.01 s > 16 s episode), so the override persists.
    if args_cli.return_home:
        rh_robot = env.unwrapped.scene["robot"]
        rh_cmd = env.unwrapped.command_manager.get_term("base_pose")
        rh_home_pos = rh_robot.data.root_pos_w.clone()                 # (E, 3)
        # phase per env: 0 outbound -> 1 turning in place toward home -> 2 returning
        rh_phase = torch.zeros(rh_home_pos.shape[0], dtype=torch.long, device=rh_home_pos.device)
        rh_timer = torch.zeros(rh_home_pos.shape[0], device=rh_home_pos.device)  # phase timer (s)
        RH_ARRIVE_DIST = 0.5   # goal-reached threshold (m), matches heading-tracking dist_thr
        RH_DWELL_S = 1.0       # stay at the goal this long before starting to turn
        RH_TURN_ERR = 0.3      # in-place turn finished when |heading error| < this (rad)
        RH_TURN_TIMEOUT = 4.0  # ... or after this long, whichever comes first

    # live hip-acceleration plot (subprocess); --vizacc only, skipped when headless
    if args_cli.vizacc and not args_cli.headless:
        hip_acc_viz_update = _build_hip_acc_plot(env)
    elif args_cli.vizacc and args_cli.headless:
        print("[INFO] --vizacc ignored: running headless.")
        hip_acc_viz_update = None
    else:
        hip_acc_viz_update = None

    obs = env.get_observations()
    
    # print observation group shapes
    print("\n" + "=" * 50)
    print("[DEBUG] Initial Observation Group Tensor Shapes:")
    for group_name, data in obs.items():
        if isinstance(data, torch.Tensor):
            print(f"  - '{group_name}': {data.shape}")
        elif isinstance(data, dict):
            print(f"  - '{group_name}' (Not Concatenated):")
            for term_name, t_data in data.items():
                print(f"      '{term_name}': {t_data.shape}")
    print("=" * 50 + "\n")

    timestep = 0
    # per-video outcome tracking (env 0): reached the commanded goal at least
    # once, and no early termination (fall) during the recording
    video_reached = False
    video_early_done = False
    sim_time = 0.0
    plot_shown = False

    if args_cli.export:
        print("[INFO] Exporting AME2 JIT policy...")
        try:
            jit_model = policy_nn.as_jit()
            jit_model.to("cpu")
            jit_model.eval()

            # script (not trace)
            traced_model = torch.jit.script(jit_model)
            traced_model = torch.jit.freeze(traced_model)
            
            iter_num = getattr(runner, "current_learning_iteration", "unknown")
            export_filename = f"ame2_exported_{iter_num}.jit"
            export_path = os.path.join(log_dir, export_filename)
            traced_model.save(export_path)
            print(f"[INFO] Successfully exported standard frozen JIT model to {export_path}")
            
            # reload the exported model for verification
            print("[INFO] Loading exported model for verification play...")
            play_model = torch.jit.load(export_path, map_location=env.unwrapped.device)
            play_model.eval()
        except Exception as e:
            print(f"[ERROR] Export failed: {e}")

    # initialize logger
    if args_cli.plot:
        logger_data = {
            "time": [],
            "base_vel": [],
            "joint_vel": [],
            "joint_torque": [],
        }
    else:
        logger_data = None

    # initialize per-term reward logger (single-episode rollout on env_idx=0)
    if args_cli.log_rewards:
        reward_manager = env.unwrapped.reward_manager
        reward_term_names = list(reward_manager.active_terms)
        print("\n" + "=" * 50)
        print(f"[INFO] LOG_REWARDS active — tracking {len(reward_term_names)} reward terms on env 0:")
        for n in reward_term_names:
            print(f"  - {n}")
        print("=" * 50 + "\n")
        reward_log = {"time": [], "total": [], **{n: [] for n in reward_term_names}}
    else:
        reward_log = None
        reward_term_names = None
    rewards_plot_done = False

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # agent stepping
            if args_cli.export and 'play_model' in locals():
                prop = obs["teacher_prop"]
                mapping = obs["teacher_mapping"]
                actions, attn_out, global_feat = play_model(prop, mapping)
                attention_weights = None
            else:
                actions, attention_weights = policy(obs)

            # --- Fly mode: external wrench lifts/levels the base and flies it to the goal ---
            if fly_controller is not None:
                fly_controller.step()
                actions *= 0.0

            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

            # --- video outcome tracking (env 0) ---
            if args_cli.video:
                vt_robot = env.unwrapped.scene["robot"]
                vt_cmd = env.unwrapped.command_manager.get_term("base_pose")
                d0 = float((vt_cmd.pos_command_w[0, :2] - vt_robot.data.root_pos_w[0, :2]).norm())
                video_reached = video_reached or d0 < 0.5
                if bool(dones.reshape(-1)[0].item()) and timestep < args_cli.video_length - 1:
                    # terminated before the last recorded step (a timeout on that step is not early)
                    video_early_done = True

            # --- return-home: goal -> (dwell) -> turn in place -> walk back ---
            if args_cli.return_home:
                done_ids = dones.reshape(-1).nonzero(as_tuple=True)[0]
                if done_ids.numel() > 0:
                    # fresh episode: record the new spawn pose, re-arm the outbound leg
                    rh_home_pos[done_ids] = rh_robot.data.root_pos_w[done_ids]
                    rh_phase[done_ids] = 0
                    rh_timer[done_ids] = 0.0

                home_vec = rh_home_pos[:, :2] - rh_robot.data.root_pos_w[:, :2]   # (E, 2)
                home_yaw = torch.atan2(home_vec[:, 1], home_vec[:, 0])

                # phase 0 -> 1: dwell 1 s inside the arrival radius (leaving resets
                # the timer), then command an in-place turn: position stays at the
                # goal, heading points at home.
                dist_goal = (rh_cmd.pos_command_w[:, :2] - rh_robot.data.root_pos_w[:, :2]).norm(dim=-1)
                at_goal = (rh_phase == 0) & (dist_goal < RH_ARRIVE_DIST)
                rh_timer = torch.where(rh_phase == 0,
                                       torch.where(at_goal, rh_timer + dt, torch.zeros_like(rh_timer)),
                                       rh_timer)
                start_turn = at_goal & (rh_timer >= RH_DWELL_S)
                ids = start_turn.nonzero(as_tuple=True)[0]
                if ids.numel() > 0:
                    rh_cmd.heading_command_w[ids] = home_yaw[ids]
                    rh_phase[ids] = 1
                    rh_timer[ids] = 0.0
                    if bool(start_turn[0].item()):
                        print("\n[return_home] env 0 at goal -> turning in place toward spawn")

                # phase 1 -> 2: once facing home (or after a timeout), walk back.
                rh_timer = torch.where(rh_phase == 1, rh_timer + dt, rh_timer)
                head_err = rh_cmd.heading_command_w - rh_robot.data.heading_w
                head_err = torch.atan2(torch.sin(head_err), torch.cos(head_err)).abs()
                start_home = (rh_phase == 1) & ((head_err < RH_TURN_ERR) | (rh_timer >= RH_TURN_TIMEOUT))
                ids = start_home.nonzero(as_tuple=True)[0]
                if ids.numel() > 0:
                    rh_cmd.pos_command_w[ids] = rh_home_pos[ids]
                    rh_cmd.heading_command_w[ids] = home_yaw[ids]   # re-aligned robot -> home
                    rh_phase[ids] = 2
                    rh_timer[ids] = 0.0
                    if bool(start_home[0].item()):
                        print("\n[return_home] env 0 facing spawn -> walking back")

            # --- Visualization ---
            env_idx = 0
            if args_cli.no_markers:
                pass  # all overlay markers disabled (clean render for videos)
            elif args_cli.viz_lidar:
                # raw lidar_cam rays for env 0: green line from the sensor origin to
                # each hit, red point at the hit; all other overlays are skipped
                lidar_draw.clear_lines()
                lidar_draw.clear_points()
                lidar = env.unwrapped.scene.sensors.get("lidar_cam", None)
                if lidar is not None:
                    hits_w = lidar.data.ray_hits_w[env_idx]      # (R, 3)
                    starts_w = lidar._ray_starts_w[env_idx]      # (R, 3)
                    dist = torch.norm(hits_w - starts_w, dim=-1)
                    # keep only rays that actually returned a hit within range
                    valid = torch.isfinite(hits_w).all(dim=-1) & (dist < lidar.cfg.max_distance + 1.0)
                    ends = hits_w[valid]
                    starts = starts_w[valid]
                    n = int(ends.shape[0])
                    if n > 0:
                        ends_l = ends.tolist()
                        starts_l = starts.tolist()
                        lidar_draw.draw_lines(starts_l, ends_l, [[0.1, 0.9, 0.1, 1.0]] * n, [1.0] * n)
                        lidar_draw.draw_points(ends_l, [[1.0, 0.2, 0.2, 1.0]] * n, [5.0] * n)
                        # sensor origin (yellow)
                        lidar_draw.draw_points([starts_w[0].tolist()], [[1.0, 1.0, 0.1, 1.0]], [12.0])
            elif is_student_task:
                # student visualization: neural map + local scan + raw point cloud
                nm_sensor = env.unwrapped.scene.sensors["neural_map"]
                # .data triggers the sensor's lazy update
                nm_data = nm_sensor.data
                if nm_data is not None:
                    # body the neural-map sensor is attached to (resolved from its prim_path)
                    robot = env.unwrapped.scene["robot"]
                    base_pos = robot.data.body_pos_w[:, nm_sensor._body_idx]
                    base_quat = robot.data.body_quat_w[:, nm_sensor._body_idx]
                    base_yaw = nm_sensor._yaw_from_q(base_quat).unsqueeze(-1) # (E, 1)
                    c = torch.cos(base_yaw)
                    s = torch.sin(base_yaw)
                    
                    # A) exported neural map, all envs
                    Xb = nm_sensor.Xb.flatten().unsqueeze(0) # (1, L*W)
                    Yb = nm_sensor.Yb.flatten().unsqueeze(0) # (1, L*W)
                    
                    Xw = c * Xb - s * Yb + base_pos[:, 0:1] # (E, L*W)
                    Yw = s * Xb + c * Yb + base_pos[:, 1:2] # (E, L*W)
                    # est is base(+cz)-relative: world_z = est + base_z + cz + 0.02 (clearance);
                    # cz is the optional em_center z offset (0 for the lidar sensor)
                    cz_off = getattr(nm_sensor.cfg, "em_center", (0.0, 0.0, 0.0))[2]
                    est = nm_data[:, :, 0] + base_pos[:, 2:3] + cz_off + 0.02 # (E, L*W)
                    unc = nm_data[:, :, 1] # (E, L*W) variance
                    
                    xyz_nm = torch.stack([Xw, Yw, est], dim=-1).reshape(-1, 3)
                    
                    # attention coloring only if the policy attends over the map cells
                    # (a teacher attends over its height scanner instead)
                    attn_matches = (
                        attention_weights is not None
                        and attention_weights.reshape(-1).numel() == xyz_nm.shape[0]
                    )
                    if attn_matches:
                        attn = attention_weights.squeeze(1).reshape(-1) # (E*504)
                        min_w = torch.min(attn)
                        max_w = torch.max(attn)
                        norm_w = (attn - min_w) / (max_w - min_w + 1e-6)
                        marker_indices = (norm_w * (num_shades - 1)).long()
                        neural_map_est_markers.visualize(translations=xyz_nm, marker_indices=marker_indices)
                    else:
                        neural_map_est_markers.visualize(translations=xyz_nm)

                    # uncertainty: cylinder height = 1-sigma std (m)
                    unc_flat = unc.reshape(-1)
                    unc_scales = torch.ones((unc_flat.shape[0], 3), device=unc.device)
                    std_flat = unc_flat.clamp_min(1e-6).sqrt()
                    unc_scales[:, 2] = torch.clamp(std_flat, min=0.01, max=1.0)

                    if attn_matches:
                        neural_map_unc_markers.visualize(translations=xyz_nm, scales=unc_scales, marker_indices=marker_indices)
                    else:
                        neural_map_unc_markers.visualize(translations=xyz_nm, scales=unc_scales)

                    # terminal readout: std stats of the queried map (viewed env)
                    std_env = unc[env_idx].clamp_min(0.0).sqrt()
                    print(f"neural_map std (env {env_idx}): "
                          f"mean {float(std_env.mean()):.3f}  min {float(std_env.min()):.3f}  "
                          f"max {float(std_env.max()):.3f}", end="\r", flush=True)
                    
                    # B) local scan grid (per-frame model input)
                    if hasattr(nm_sensor, "_sensor_z_local"):
                        z_local = nm_sensor._sensor_z_local # (E, Ls, Ws)
                        B, Ls, Ws = z_local.shape
                        em_cfg = nm_sensor.cfg
                        if hasattr(em_cfg, "em_center"):
                            # grid around the base, heights base(+cz)-relative, all envs
                            cx, cy, cz = em_cfg.em_center
                            xs_em = torch.linspace(-em_cfg.em_length/2, em_cfg.em_length/2, Ls, device=z_local.device) + cx
                            ys_em = torch.linspace(-em_cfg.em_width/2, em_cfg.em_width/2, Ws, device=z_local.device) + cy
                            Xem, Yem = torch.meshgrid(xs_em, ys_em, indexing='ij')
                            Xem_f = Xem.flatten().unsqueeze(0)
                            Yem_f = Yem.flatten().unsqueeze(0)

                            Xw_em = c * Xem_f - s * Yem_f + base_pos[:, 0:1]
                            Yw_em = s * Xem_f + c * Yem_f + base_pos[:, 1:2]
                            Zw_em = z_local.reshape(B, -1) + cz + base_pos[:, 2:3]

                            xyz_em = torch.stack([Xw_em, Yw_em, Zw_em], dim=-1).reshape(-1, 3)
                        else:
                            # lidar sensor: grid around the lidar origin (base-yaw axes),
                            # lidar-z-relative heights, holes = -3; viewed env only, holes masked
                            lidar = env.unwrapped.scene.sensors[em_cfg.lidar_sensor_name]
                            lidar_pos = lidar.data.pos_w  # (E, 3)
                            Xem_f = nm_sensor.Xs.flatten().unsqueeze(0) # (1, Ls*Ws)
                            Yem_f = nm_sensor.Ys.flatten().unsqueeze(0)

                            Xw_em = c * Xem_f - s * Yem_f + lidar_pos[:, 0:1]
                            Yw_em = s * Xem_f + c * Yem_f + lidar_pos[:, 1:2]
                            Zw_em = z_local.reshape(B, -1) + lidar_pos[:, 2:3]

                            xyz_em = torch.stack([Xw_em, Yw_em, Zw_em], dim=-1)[env_idx]
                            xyz_em = xyz_em[z_local[env_idx].reshape(-1) > -2.9]
                        local_scan_markers.visualize(translations=xyz_em)

                    # C) raw lidar point cloud
                    if hasattr(nm_sensor, "_raw_hits_w"):
                        raw_hits = nm_sensor._raw_hits_w
                        # hits are tagged per env; draw only the viewed env
                        if hasattr(nm_sensor, "_raw_hits_env"):
                            raw_hits = raw_hits[nm_sensor._raw_hits_env == env_idx]
                        if raw_hits.shape[0] > 0:
                            raw_pcl_markers.visualize(translations=raw_hits)
            else:
                # teacher visualization: critic/actor heights + attention
                # A. critic
                if hasattr(height_scanner_critic_sensor.data, "_processed_z_w"):
                    z_critic_w = height_scanner_critic_sensor.data._processed_z_w
                    hit_pos_critic = height_scanner_critic_sensor._ray_starts_w.clone() # (E, R, 3)
                    hit_pos_critic[..., 2] = z_critic_w
                    critic_markers.visualize(translations=hit_pos_critic[env_idx])

                # B. actor processed heights and attention
                if hasattr(height_scanner_sensor.data, "_processed_z_w"):
                    z_actor_w = height_scanner_sensor.data._processed_z_w
                    hit_pos_actor = height_scanner_sensor._ray_starts_w.clone() # (E, R, 3)
                    hit_pos_actor[..., 2] = z_actor_w
                    actor_markers.visualize(translations=hit_pos_actor[env_idx])
                    
                    if attention_weights is not None:
                        if attention_weights.ndim == 3:
                            attention_weights = attention_weights.squeeze(1)
                        min_weights = torch.min(attention_weights, dim=-1, keepdim=True)[0]
                        max_weights = torch.max(attention_weights, dim=-1, keepdim=True)[0]
                        norm_weights = (attention_weights - min_weights) / (max_weights - min_weights + 1e-3)
                        marker_indices = (norm_weights * (num_shades - 1)).long()
                        attention_markers.visualize(
                            translations=hit_pos_actor[env_idx], 
                            marker_indices=marker_indices[env_idx]
                        )
            
            # hip acceleration plot
            if hip_acc_viz_update is not None:
                hip_acc_viz_update(env_idx)

            if args_cli.log_rewards and not rewards_plot_done:
                # env-0 per-term reward (reward/s, as in RewardManager._step_reward)
                step_rew = env.unwrapped.reward_manager._step_reward[0].detach().cpu().numpy()
                reward_log["time"].append(sim_time)
                reward_log["total"].append(float(step_rew.sum()))
                for i, name in enumerate(reward_term_names):
                    reward_log[name].append(float(step_rew[i]))
                # plot when env 0 terminates; only the first episode is captured
                if bool(dones[0].item()):
                    print(f"[INFO] LOG_REWARDS: env 0 terminated at t={sim_time:.2f}s after {len(reward_log['time'])} steps. Plotting…")
                    _plot_reward_curves(reward_log, reward_term_names, log_dir, dt)
                    rewards_plot_done = True

            if args_cli.plot:
                # log data for debug plots
                robot = env.unwrapped.scene["robot"]
                logger_data["time"].append(sim_time)
                logger_data["base_vel"].append(robot.data.root_lin_vel_b[0].cpu().numpy().copy())
                logger_data["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
                logger_data["joint_torque"].append(robot.data.applied_torque[0].cpu().numpy().copy())
                
                # plot after 10 s of sim time
                if sim_time >= 10.0 and not plot_shown:
                    print(f"[INFO] 10 seconds reached. Generating non-blocking debug plots...")
                    if len(logger_data["time"]) > 0:
                        time_arr = np.array(logger_data["time"])
                        base_vel = np.array(logger_data["base_vel"])
                        joint_vel = np.array(logger_data["joint_vel"])
                        joint_torque = np.array(logger_data["joint_torque"])

                        num_joints = joint_vel.shape[1]
                        num_cols = min(4, num_joints)
                        num_rows = math.ceil(num_joints / num_cols) + 1
                        
                        fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
                        
                        # base velocity (top row)
                        ax_base = plt.subplot2grid((num_rows, num_cols), (0, 0), colspan=num_cols)
                        ax_base.plot(time_arr, base_vel[:, 0], label="Base Vel X")
                        ax_base.plot(time_arr, base_vel[:, 1], label="Base Vel Y")
                        ax_base.plot(time_arr, base_vel[:, 2], label="Base Vel Z")
                        ax_base.set_title("Base Velocity [Body Frame] vs Time")
                        ax_base.set_xlabel("Time [s]")
                        ax_base.set_ylabel("Velocity [m/s]")
                        ax_base.legend()
                        ax_base.grid(True)

                        # per-joint torque vs. velocity
                        joint_names = robot.data.joint_names if hasattr(robot.data, "joint_names") else [f"Joint {i}" for i in range(num_joints)]
                        
                        for i in range(num_joints):
                            grid_row = (i // num_cols) + 1
                            grid_col = i % num_cols
                            ax = plt.subplot2grid((num_rows, num_cols), (grid_row, grid_col))
                            
                            ax.scatter(joint_vel[:, i], joint_torque[:, i], alpha=0.5, s=2)
                            ax.set_title(f"{joint_names[i]}")
                            ax.set_xlabel("Vel [rad/s]")
                            ax.set_ylabel("Torque [Nm]")
                            ax.grid(True)

                        plt.tight_layout()
                        
                        plot_path = os.path.join(log_dir, "debug_plots.pdf")
                        plt.savefig(plot_path)
                        print(f"[INFO] Saved detailed debug plots to: {plot_path}")
                        
                        plt.close(fig)
                        plot_shown = True

            sim_time += dt

        if args_cli.video:
            timestep += 1
            # exit after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.video:
        # machine-readable clip outcome (parsed by video-rendering drivers)
        print(f"[PLAY_RESULT] reached={video_reached} early_done={video_early_done} "
              f"success={video_reached and not video_early_done}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
