#!/usr/bin/env bash
set -euo pipefail

readonly CONTAINER_NAME="road-detection-gpu"
readonly IMAGE="pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime"

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    docker start "${CONTAINER_NAME}" >/dev/null
else
    docker run --detach \
        --name "${CONTAINER_NAME}" \
        --gpus all \
        --ipc host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --volume "${PROJECT_DIR}:/workspace" \
        --workdir /workspace \
        "${IMAGE}" \
        sleep infinity >/dev/null
fi

echo "Container ${CONTAINER_NAME} is running."
echo "Open a shell: docker exec -it ${CONTAINER_NAME} bash"
echo "Check CUDA:  docker exec ${CONTAINER_NAME} python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
