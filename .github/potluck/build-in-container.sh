#!/usr/bin/env bash
# Runs inside the pinned build image (see build.sh). Installs the pinned
# packages from the Ubuntu archive snapshot, configures and builds, and writes
# the binaries plus build-info.json to /out/$PLATFORM.
set -euo pipefail

# shellcheck source=toolchain.env
. /src/.github/potluck/toolchain.env

case "$PLATFORM" in
  linux-amd64-vulkan) snapshot=$VULKAN_APT_SNAPSHOT; packages=$VULKAN_PACKAGES; flags=$VULKAN_CMAKE_FLAGS ;;
  linux-amd64-cuda)   snapshot=$CUDA_APT_SNAPSHOT;   packages=$CUDA_PACKAGES;   flags=$CUDA_CMAKE_FLAGS ;;
  *) echo "unknown platform: $PLATFORM" >&2; exit 2 ;;
esac

export DEBIAN_FRONTEND=noninteractive
# Only the Ubuntu archive is used. The CUDA image also lists NVIDIA's apt
# source; nothing is installed from it, so it is dropped.
rm -f /etc/apt/sources.list.d/cuda*.list
# The snapshot service is HTTPS only: certificates come from the live archive
# first (the only unpinned step), then every package from the snapshot.
apt-get update -qq
apt-get install -y -qq --no-install-recommends ca-certificates >/dev/null
apt-get update -qq --snapshot "$snapshot"
# shellcheck disable=SC2086
apt-get install -y -qq --no-install-recommends $packages >/dev/null

git config --global --add safe.directory /src

read -r -a cmake_flags <<<"$flags"
cmake -S /src -B /build -G Ninja "${cmake_flags[@]}"
nice -n "$BUILD_NICE" cmake --build /build --target llama-server ggml-rpc-server -j "$JOBS"

dest=/out/$PLATFORM
rm -rf "$dest"
mkdir -p "$dest"
install -m 0755 /build/bin/llama-server "$dest/llama-server"
install -m 0755 /build/bin/ggml-rpc-server "$dest/rpc-server"

# --version needs the runtime libraries. For CUDA the driver library is not in
# the image (no GPU here): the toolkit's stub stands in for it just long enough
# to print the version.
version_out=""
if [ "$PLATFORM" = linux-amd64-cuda ]; then
  mkdir -p /tmp/cudastub
  ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cudastub/libcuda.so.1
  version_out=$(LD_LIBRARY_PATH=/tmp/cudastub "$dest/llama-server" --version 2>&1 || true)
else
  version_out=$("$dest/llama-server" --version 2>&1 || true)
fi

needed() { objdump -p "$1" | awk '/NEEDED/ {print $2}' | sort -u; }
glibc_min() { objdump -T "$1" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1; }
pkgver() { dpkg-query -W -f='${Version}' "$1" 2>/dev/null || echo ""; }

nvcc_version=""
if command -v nvcc >/dev/null; then
  nvcc_version=$(nvcc --version | sed -n 's/.*release \([0-9.]*\), V\([0-9.]*\).*/\2/p')
fi

python3 --version >/dev/null 2>&1 || apt-get install -y -qq --no-install-recommends python3 >/dev/null

PLATFORM="$PLATFORM" IMAGE="$IMAGE" SNAPSHOT="$snapshot" FLAGS="$flags" \
VERSION_OUT="$version_out" NVCC_VERSION="$nvcc_version" \
GCC_VERSION="$(gcc -dumpfullversion)" CMAKE_VERSION="$(cmake --version | head -1 | awk '{print $3}')" \
NINJA_VERSION="$(ninja --version)" GLIBC_BUILD="$(ldd --version | head -1 | awk '{print $NF}')" \
VULKAN_LOADER="$(pkgver libvulkan-dev)" GLSLC="$(pkgver glslc)" SPIRV_HEADERS="$(pkgver spirv-headers)" \
CUDA_CUDART="$(pkgver cuda-cudart-dev-13-3)" CUBLAS="$(pkgver libcublas-dev-13-3)" NCCL="$(pkgver libnccl-dev)" \
NEEDED_SERVER="$(needed "$dest/llama-server" | tr '\n' ' ')" NEEDED_RPC="$(needed "$dest/rpc-server" | tr '\n' ' ')" \
GLIBC_SERVER="$(glibc_min "$dest/llama-server")" GLIBC_RPC="$(glibc_min "$dest/rpc-server")" \
python3 - <<'EOF' >"$dest/build-info.json"
import json, os, re
e = os.environ
def ver(a, b):
    pa = [int(x) for x in a.split('.')]
    pb = [int(x) for x in b.split('.')]
    return a if pa >= pb else b
libs = sorted(set(e["NEEDED_SERVER"].split()) | set(e["NEEDED_RPC"].split()))
toolchain = {
    "image": e["IMAGE"],
    "apt_snapshot": e["SNAPSHOT"],
    "gcc": e["GCC_VERSION"],
    "cmake": e["CMAKE_VERSION"],
    "ninja": e["NINJA_VERSION"],
    "build_glibc": e["GLIBC_BUILD"],
}
if e["PLATFORM"].endswith("-vulkan"):
    toolchain.update({"vulkan_loader_dev": e["VULKAN_LOADER"], "glslc": e["GLSLC"], "spirv_headers": e["SPIRV_HEADERS"]})
else:
    toolchain.update({"cuda_nvcc": e["NVCC_VERSION"], "cuda_cudart_dev": e["CUDA_CUDART"], "cublas_dev": e["CUBLAS"], "nccl_dev": e["NCCL"]})
m = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", e["VERSION_OUT"])
print(json.dumps({
    "platform": e["PLATFORM"],
    "cmake_flags": e["FLAGS"].split(),
    "toolchain": toolchain,
    "version_output": e["VERSION_OUT"].strip(),
    "build_number": int(m.group(1)) if m else None,
    "build_commit_short": m.group(2) if m else None,
    "needs": {"glibc_min": ver(e["GLIBC_SERVER"], e["GLIBC_RPC"]), "libraries": libs},
}, indent=2))
EOF

chown -R "$HOST_UID:$HOST_GID" "$dest"
cat "$dest/build-info.json"
