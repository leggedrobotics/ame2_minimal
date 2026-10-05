#!/bin/bash
# Build the AME2 image.   IMAGE=ame2-minimal:latest BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2 ./build.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
IMAGE=${IMAGE:-ame2-minimal:latest}
BASE_IMAGE=${BASE_IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.2}
DOCKER_BUILDKIT=1 docker build -f "$HERE/Dockerfile" --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$IMAGE" "$ROOT"
echo "[build] done -> $IMAGE"
