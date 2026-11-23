"""Record the student's lidar neural map against the ground-truth terrain during play (env 0).

Used by ``play.py --record_nmap OUT.npz``; render the file with ``render_neural_map.py``.

Per recorded step it stores, on a base-yaw-aligned grid around the robot (``x_range`` x ``y_range`` at ``res``):
the fused neural-map height and variance (queried from the sensor's global map, finer and larger than the
policy's 26 x 15 export), the true terrain height (ray cast straight down onto the terrain mesh), the latest
lidar returns near the robot, the robot link positions and the commanded goal.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.sensors import RayCaster
from isaaclab.utils.warp import raycast_mesh


class NeuralMapRecorder:
    def __init__(self, env, path: str, every: int = 2, x_range=(-1.5, 3.0), y_range=(-1.5, 1.5), res: float = 0.05,
                 max_hits: int = 4000, env_idx: int = 0):
        self.env = env.unwrapped
        self.path, self.every, self.env_idx, self.max_hits = path, every, env_idx, max_hits
        self.sensor = self.env.scene.sensors["neural_map"]
        self.lidar = self.env.scene.sensors[self.sensor.cfg.lidar_sensor_name]
        self.robot = self.env.scene["robot"]
        dev = self.sensor._device
        xs = torch.arange(x_range[0], x_range[1] + 1e-6, res, device=dev)
        ys = torch.arange(y_range[0], y_range[1] + 1e-6, res, device=dev)
        self.Xg, self.Yg = torch.meshgrid(xs, ys, indexing="ij")
        self.mesh = RayCaster.meshes[self.lidar.cfg.mesh_prim_paths[0]]
        self.frames: dict[str, list] = {k: [] for k in (
            "t", "base_pos", "base_yaw", "body_pos", "est", "var", "gt", "hits", "goal", "done")}
        self.body_names = list(self.robot.body_names)
        self.meta = dict(x_range=np.array(x_range), y_range=np.array(y_range), res=res,
                         dt=float(self.env.step_dt) * every)
        self._step = 0

    @torch.inference_mode()
    def step(self, dones: torch.Tensor):
        self._step += 1
        if self._step % self.every:
            return
        i = self.env_idx
        _ = self.sensor.data  # lazy update: integrates any due scan, sets the query pose
        base_pos = self.robot.data.body_pos_w[:, self.sensor._body_idx]
        base_yaw = self.sensor._yaw_from_q(self.robot.data.body_quat_w[:, self.sensor._body_idx])
        mean, var = self.sensor.pipeline.export_to_base(self.Xg, self.Yg)  # base-z-relative, (E, 1, L, W)
        c, s = torch.cos(base_yaw[i]), torch.sin(base_yaw[i])
        xw = c * self.Xg - s * self.Yg + base_pos[i, 0]
        yw = s * self.Xg + c * self.Yg + base_pos[i, 1]
        starts = torch.stack([xw, yw, torch.full_like(xw, 50.0)], -1).reshape(1, -1, 3)
        dirs = torch.zeros_like(starts); dirs[..., 2] = -1.0
        gt = raycast_mesh(starts.contiguous(), dirs.contiguous(), max_dist=200.0, mesh=self.mesh)[0][0, :, 2]
        hits = self.lidar.data.ray_hits_w[i]
        hits = hits[torch.isfinite(hits).all(-1)]
        rel = hits[:, :2] - base_pos[i, :2]
        hits = hits[(rel.abs() < 3.5).all(-1)]
        if hits.shape[0] > self.max_hits:
            hits = hits[torch.randperm(hits.shape[0], device=hits.device)[: self.max_hits]]
        f = self.frames
        f["t"].append(self._step * float(self.env.step_dt))
        f["base_pos"].append(base_pos[i].cpu().numpy())
        f["base_yaw"].append(float(base_yaw[i]))
        f["body_pos"].append(self.robot.data.body_pos_w[i].cpu().numpy())
        f["est"].append((mean[i, 0] + base_pos[i, 2]).cpu().numpy().astype(np.float32))  # world z
        f["var"].append(var[i, 0].cpu().numpy().astype(np.float32))
        f["gt"].append(gt.reshape(self.Xg.shape).cpu().numpy().astype(np.float32))
        f["hits"].append(hits.cpu().numpy().astype(np.float32))
        f["goal"].append(self.env.command_manager.get_term("base_pose").pos_command_w[i].cpu().numpy())
        f["done"].append(bool(dones.reshape(-1)[i]))

    def save(self):
        f = self.frames
        hits_len = np.array([len(h) for h in f["hits"]])
        out = {k: np.stack(v) for k, v in f.items() if k != "hits"}
        out["hits"] = np.concatenate(f["hits"]) if f["hits"] else np.zeros((0, 3), np.float32)
        out["hits_len"] = hits_len
        out["body_names"] = np.array(self.body_names)
        np.savez_compressed(self.path, **out, **self.meta)
        print(f"[NMAP] saved {len(f['t'])} frames to {self.path}")
