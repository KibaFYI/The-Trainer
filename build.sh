#!/usr/bin/env bash
# build (and optionally push) the image
#   ./build.sh                       -> axolotl-forge:latest
#   ./build.sh you/axolotl-forge     -> builds you/axolotl-forge:0.21.1-cu130 + :latest
#   PUSH=1 ./build.sh you/axolotl-forge
#   WITH_FA4=1 WITH_DEEPEP=1 DEEPEP_ARCH=9.0 ./build.sh you/axolotl-forge   (Hopper extras)
set -euo pipefail
IMAGE="${1:-axolotl-forge}"
AXOLOTL_VERSION="${AXOLOTL_VERSION:-0.21.1}"
TAG="${AXOLOTL_VERSION}-cu130"
cd "$(dirname "$0")"
docker buildx build --platform linux/amd64 \
    --build-arg AXOLOTL_VERSION="$AXOLOTL_VERSION" \
    --build-arg WITH_LLMCOMPRESSOR="${WITH_LLMCOMPRESSOR:-1}" \
    --build-arg WITH_FA4="${WITH_FA4:-0}" \
    --build-arg WITH_DEEPEP="${WITH_DEEPEP:-0}" \
    --build-arg DEEPEP_ARCH="${DEEPEP_ARCH:-9.0}" \
    --build-arg MAX_JOBS="${MAX_JOBS:-8}" \
    -t "$IMAGE:$TAG" -t "$IMAGE:latest" \
    $([ "${PUSH:-0}" = "1" ] && echo --push || echo --load) .
echo "built $IMAGE:$TAG"
