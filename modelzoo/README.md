# Model zoo

Checkpoints and their training configs (`params/agent.yaml`, `params/env.yaml`), with the WandB report of each run.

> **Note:** these models simply prove that the pipeline works. They are not fully converged; longer training gives better policies.

| Model | Task | Training | Checkpoint | Report |
|---|---|---|---|---|
| [`g1_gaze_64000env_10h`](g1_gaze_64000env_10h) | `Ame2-G1-Gaze` (teacher) | 64000 envs (4 GPUs x 16000), 2400 iterations, ~10 h | `model_2400.pt` | [WandB](https://forge.coreweave.com/wandb/zitac/ame2_os/reports/g1_gaze_64000env_10h--VmlldzoxODA2MzM2Mg?accessToken=opbrkcokw5i2scft1uzi0t4nn0v408n7zoqdbcsarbqu3jphs2b7lz9v3os4yfu5) |
| [`g1_gaze_student_16000env_1000it`](g1_gaze_student_16000env_1000it) | `Ame2-G1-Gaze-Student` (student) | 16000 envs (1 GPU), 1000 iterations (500 distillation + 500 distillation & RL), ~8 h | `model_999.pt` | [WandB](https://forge.coreweave.com/wandb/zitac/ame2_os/reports/ame2-gaze-g1-student--VmlldzoxODM1NDM5Mw?accessToken=fk4vq9e8nvao9px4azh4xiefyn4orbwbdhe7rb6b6a6vjeevxz25s0maq9jftcwe) |

## g1_gaze_64000env_10h

AME-2 G1 teacher with TAGA-style active gaze, trained on the privileged height scan (stability check of the
open-source training setup). Its actor, exported with `play.py --export`, is the frozen distillation teacher of
the student: [`ame2/ame2/NN_models/g1_gaze_teacher_isaaclab.jit`](../ame2/ame2/NN_models/g1_gaze_teacher_isaaclab.jit).

## g1_gaze_student_16000env_1000it

G1 student of the teacher above. It walks on the lidar neural map (Mid360 scans with the self-body and 0.5 m
near-range filters, fused by the pretrained mapping model) plus a 20-step proprioception history, with the teacher's
gaze crop. Iterations 0-499 are pure distillation, then RL is added (`rl_switch_point=501`) with a multi-headed
critic.

At the end of training: goal-reaching success 0.70 (EMA), 0.77 averaged over the 12 terrain types, at mean
terrain level 6.3 of 9.

## Usage

The container mounts `modelzoo/` read-only, so play a checkpoint as is, or copy it under `ame2/logs/` first for
`--export` / `--video` (they write next to the checkpoint):

```bash
./container/run.sh -g 0 --gui play --task Ame2-G1-Gaze-Play --checkpoint ../modelzoo/g1_gaze_64000env_10h/model_2400.pt
./container/run.sh -g 0 --gui play --task Ame2-G1-Gaze-Student-Play --checkpoint ../modelzoo/g1_gaze_student_16000env_1000it/model_999.pt
```
