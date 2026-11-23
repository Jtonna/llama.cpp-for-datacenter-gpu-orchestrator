#!/usr/bin/env bash
# Build llama-server and rpc-server for one Potluck platform inside the pinned
# build container. Used by .github/workflows/release.yml and for local checks.
#
#   .github/potluck/build.sh <platform> <out-dir>
#
#   platform: linux-amd64-vulkan | linux-amd64-cuda
#   out-dir:  receives <platform>/llama-server, <platform>/rpc-server and
#             <platform>/build-info.json
#
# Environment:
#   JOBS        parallel build jobs (default: number of CPUs)
#   BUILD_NICE  niceness of the compiler processes (default 0)
#
# The source tree must be a git checkout with full history (the build number
# printed by --version is the commit count). It is mounted read-only; the build
# directory lives inside the container. No GPU is given to the container.
set -euo pipefail

platform=${1:?platform}
out=${2:?out-dir}
here=$(cd "$(dirname "$0")" && pwd)
src=$(cd "$here/../.." && pwd)

# shellcheck source=toolchain.env
. "$here/toolchain.env"

case "$platform" in
  linux-amd64-vulkan) image=$VULKAN_IMAGE ;;
  linux-amd64-cuda)   image=$CUDA_IMAGE ;;
  *) echo "unknown platform: $platform" >&2; exit 2 ;;
esac

git -C "$src" rev-parse --verify HEAD >/dev/null
if [ "$(git -C "$src" rev-parse --is-shallow-repository)" = true ]; then
  echo "the checkout is shallow; the build number would be wrong (use fetch-depth: 0)" >&2
  exit 1
fi

mkdir -p "$out"
out=$(cd "$out" && pwd)

docker pull --quiet "$image"
docker run --rm \
  --network bridge \
  -v "$src:/src:ro" \
  -v "$out:/out" \
  -e PLATFORM="$platform" \
  -e IMAGE="$image" \
  -e JOBS="${JOBS:-$(nproc)}" \
  -e BUILD_NICE="${BUILD_NICE:-0}" \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  "$image" bash /src/.github/potluck/build-in-container.sh
