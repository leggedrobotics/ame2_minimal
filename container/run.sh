#!/bin/bash
# Run AME2 training / play in the local container (multi-GPU training via torchrun).
#
#   ./run.sh [-g GPUS] [--gui] [-d] train <train.py args>   # e.g. -g 0,1 train --task Ame2-G1-Gaze --num_envs 4096
#   ./run.sh [-g GPU]  [--gui]      play  <play.py args>    # single GPU
#   ./run.sh [-g GPUS]              shell                   # new container, interactive bash
#   ./run.sh [-g GPUS]  -d          shell                   # long-lived background container (for exec)
#   ./run.sh exec [NAME]                                    # bash into a running container (default: newest ame2_*)
#
#   -g, --gpus   host GPU ids, comma-separated (default 0); >1 id = torchrun + --distributed
#   --gui        no --headless, forward X11 (needs $DISPLAY and `xhost +local:` on the host)
#   -d, --detach run in the background (follow with `docker logs -f <name>`)
#
# Env: IMAGE (default ame2-minimal:latest), WANDB_API_KEY (passed through if set),
#      SHM_SIZE (default 32g), MASTER_PORT (default 29500).
# Logs land in ame2/logs/ on the host (the repo is bind-mounted).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
IMAGE=${IMAGE:-ame2-minimal:latest}
GPUS=0; GUI=0; DETACH=0
while [ $# -gt 0 ]; do
    case "$1" in
        -g|--gpus)   GPUS="$2"; shift 2 ;;
        --gui)       GUI=1; shift ;;
        -d|--detach) DETACH=1; shift ;;
        -h|--help)   sed -n '2,15p' "$0"; exit 0 ;;
        *) break ;;
    esac
done
[ $# -ge 1 ] || { sed -n '2,15p' "$0"; exit 1; }
MODE=$1; shift

if [ "$MODE" = exec ]; then
    NAME=${1:-$(docker ps --filter "name=^ame2_" --format '{{.Names}}' | head -1)}
    [ -n "$NAME" ] || { echo "[run] no running ame2_* container (start one with: ./run.sh -d shell)" >&2; exit 1; }
    echo "[run] exec -> $NAME"
    if [ -t 0 ]; then exec docker exec -it -w /workspace/ame2-minimal/ame2 "$NAME" bash
    else exec docker exec -i -w /workspace/ame2-minimal/ame2 "$NAME" bash; fi
fi
IFS=',' read -ra GPU_ARR <<< "$GPUS"; NGPU=${#GPU_ARR[@]}

PY=/isaac-sim/python.sh
HEADLESS=--headless; [ "$GUI" = 1 ] && HEADLESS=
case "$MODE" in
    train)
        if [ "$NGPU" -gt 1 ]; then
            CMD="$PY -m torch.distributed.run --nnodes=1 --nproc_per_node=$NGPU --master_port=${MASTER_PORT:-29500} scripts/rsl_rl/train.py --distributed $HEADLESS"
        else
            CMD="$PY scripts/rsl_rl/train.py $HEADLESS"
        fi ;;
    play)
        [ "$NGPU" -gt 1 ] && { echo "[run] play uses one GPU: ${GPU_ARR[0]}"; GPUS=${GPU_ARR[0]}; }
        CMD="$PY scripts/rsl_rl/play.py $HEADLESS" ;;
    shell) CMD="bash"; [ "$DETACH" = 1 ] && CMD="sleep infinity" ;;
    *) echo "[run] unknown mode '$MODE' (train|play|shell|exec)" >&2; exit 1 ;;
esac
[ "$MODE" != shell ] && [ $# -gt 0 ] && CMD="$CMD$(printf ' %q' "$@")"

NAME="ame2_${MODE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT/ame2/logs"
RUN=(docker run --name "$NAME" --gpus "\"device=$GPUS\"" --network host
    --shm-size "${SHM_SIZE:-32g}" --ulimit memlock=-1 --ulimit stack=67108864
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -v "$ROOT/ame2:/workspace/ame2-minimal/ame2:rw"
    -v "$ROOT/rsl_rl:/workspace/ame2-minimal/rsl_rl:rw"
    -v "$ROOT/modelzoo:/workspace/ame2-minimal/modelzoo:ro"
    -v ame2-cache-kit:/isaac-sim/kit/cache
    -v ame2-cache-ov:/root/.cache/ov
    -v ame2-cache-pip:/root/.cache/pip
    -v ame2-cache-gl:/root/.cache/nvidia/GLCache
    -v ame2-cache-compute:/root/.nv/ComputeCache
    -v ame2-omni-logs:/root/.nvidia-omniverse/logs
    -v ame2-omni-data:/root/.local/share/ov/data
    -w /workspace/ame2-minimal/ame2)
[ -n "${WANDB_API_KEY:-}" ] && RUN+=(-e WANDB_API_KEY)
if [ "$GUI" = 1 ]; then
    RUN+=(-e "DISPLAY=${DISPLAY:-:0}" -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
fi
if [ "$DETACH" = 1 ]; then RUN+=(-d)
elif [ -t 0 ]; then RUN+=(--rm -it)
else RUN+=(--rm -i); fi
RUN+=("$IMAGE" bash -lc "$CMD")

echo "[run] gpus $GPUS -> container $NAME"
echo "[run] $CMD"
exec "${RUN[@]}"
