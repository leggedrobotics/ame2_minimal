# Minimal AME-2 extension of IsaacLab

<img src="misc/all_terrains_4x3.webp" width="100%" alt="AME-2 G1 on all training terrains">

Website: https://sites.google.com/leggedrobotics.com/ame-2  
Result Video: https://www.youtube.com/watch?v=FNA1crvtBLs  
Follow-up work: the AME-2 map encoder and the lidar neural map nicely support our new work [ACP: Humanoid Mobility via Attention-based Contacts](https://leggedrobotics.github.io/ACP/).  
Author: Chong Zhang (chozhang@ethz.ch or chong.zhang@ai.ethz.ch)

Please cite if you find this codebase or our paper useful:
```bibtex
@article{zhang2026agile,
  title   = {Agile and Generalized Legged Locomotion via Attention-Based Neural Map Encoding},
  author  = {Zhang, Chong and Klemm, Victor and Yang, Fan and Hutter, Marco},
  journal = {IEEE Transactions on Robotics},
  year    = {2026}
}
```

The codebase is mostly vibecoded, and many IsaacLab know-hows are from Mayank Mittal. Thanks AI and Mayank! 

> **Vibe4Vibe:** the fastest way to get into this codebase is a coding agent.

> **No feature contribution:** It shall be half-archived as a reference to a finished work.


## Setup

+ Isaac Lab 2.3.2 (Isaac Sim 5.1)
+ rsl_rl: bundled in `rsl_rl/`
+ install via container (from the repo root, needs Docker + NVIDIA Container Toolkit):
  ```bash
  ./container/build.sh                                                        # build ame2-minimal:latest
  ./container/run.sh -g 0 train --task Ame2-G1-Gaze --num_envs 4096           # single-GPU training
  ./container/run.sh -g 0,1 train --task Ame2-G1-Gaze --num_envs 4096         # multi-GPU (torchrun)
  ./container/run.sh -g 0 --gui play --task Ame2-G1-Gaze-Play --checkpoint <path/to/model.pt>
  ./container/run.sh shell                                                    # interactive shell
  ./container/run.sh -d shell                                                 # background container to exec into
  ./container/run.sh exec [NAME]                                              # bash into a running container (default: newest)
  ```




## Current features:
- G1 teacher training (AME-2, optional TAGA-style active gaze, multi-headed PPO, bf16 optimization, muon optimizer)
- Neural Mapping in the loop (Mid360 lidar with self-body and near-range filters, pretrained mapping model in `ame2/ame2/NN_models/`)
- G1 student training (LSIO + AME-2, distilled + RL from a teacher JIT, multi-headed critic)

- Optional new features:
    - torso, arm, and foot shaping for humanoids, not tuned. The original paper only covers legs.

## Neural map in the loop:
The student's lidar neural map (coloured cells) against the true terrain (grey mesh) during rollouts of the
[model-zoo student](modelzoo/README.md), with the current Mid360 scan in red. Recorded with
`play.py --record_nmap` and rendered with `ame2/scripts/rsl_rl/render_neural_map.py`.

<img src="misc/neural_map_4terrains.webp" width="100%" alt="Lidar neural map vs. true terrain on stairs, a climb, stepping stones and boxes">

## Existing checkpoints and logs:
- Check [`modelzoo/`](modelzoo/README.md): the G1 gaze teacher and its student. Will add new runs in the future.

## Release Plans:
- Neural Map Training Code (ETA: late December 2026)

## Notes:
- Some trivial details might not be perfectly aligned between IsaacLab and legged gym. Some might be improved.
- The results for humanoids do not represent the limits of our method (contributions mostly on generalization and neural mapping) -- there is no targetted performance tuning or style optimization, especially for G1.
- MDP setup shall easily transfer to other robots.

## License

© 2026 ETH Zurich  
Created by: Chong Zhang, Victor Klemm, Fan Yang, and Marco Hutter  
Licensed under the GNU General Public License v3.0 (see [LICENSE](LICENSE)).

Third-party code keeps its original license (see [licenses/dependencies](licenses/dependencies)):
- `rsl_rl/` is a modified copy of [rsl_rl](https://github.com/leggedrobotics/rsl_rl), BSD-3-Clause.
- Files in `ame2/` marked "The Isaac Lab Project Developers" are derived from [Isaac Lab](https://github.com/isaac-sim/IsaacLab), BSD-3-Clause. Isaac Lab itself is not included.
- `ame2/ame2/sensors/mid360_raydirs.npy` (Livox Mid-360 scan pattern) comes from [fratopa/Mid360_simulation_plugin](https://github.com/fratopa/Mid360_simulation_plugin).

