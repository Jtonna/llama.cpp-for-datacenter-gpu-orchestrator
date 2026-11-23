# Potluck llama builds

This fork publishes its own builds of `llama-server` and `rpc-server` for
[Potluck](https://github.com/Jtonna/datacenter-gpu-orchestrator), the scheduler
that runs models across a pool of machines. A release is made by
`.github/workflows/release.yml` when a tag `potluck-llama-v*` is pushed. Nothing
is built or published by hand.

## What a release contains

| Asset | What it is |
|---|---|
| `potluck-llama-<tag>-linux-amd64-vulkan.tar.gz` | Machines with an AMD (or other Vulkan) GPU |
| `potluck-llama-<tag>-linux-amd64-cuda.tar.gz` | Machines with an Nvidia GPU |
| `potluck-llama-manifest.json` | What was built, from what, with what, and the checksum and size of every file |
| `SHA256SUMS` | sha256 of every other asset, in `sha256sum` format |

Each archive unpacks to:

```
LICENSE-llama.cpp
<platform>/
  llama-server
  rpc-server
  SHA256SUMS        sha256sum output for the two programs
```

The platform folder has the layout Potluck serves from its programs folder
(`programs/<release>/<platform>/`), so it can be copied there unchanged.

### The programs

- `llama-server` is the host program, `rpc-server` the worker program. Both come
  from the tagged commit and print it: `llama-server --version` answers
  `version: 1.0.0-dev (build <commit count>, commit <short commit>)`.
- Each is a single file: llama and ggml are linked statically
  (`BUILD_SHARED_LIBS=OFF`), the binaries are not stripped, and no shared
  library is shipped beside them. They use what the machine provides:
  - `linux-amd64-vulkan`: the C and C++ runtime, `libgomp.so.1` (GNU OpenMP)
    and `libvulkan.so.1` (the Vulkan loader, with a driver such as Mesa RADV).
    Built on Ubuntu 26.04, so they need glibc 2.43 or newer.
  - `linux-amd64-cuda`: the C and C++ runtime, `libgomp.so.1`, the Nvidia
    driver (`libcuda.so.1`) and the CUDA 13 runtime libraries `libcudart.so.13`,
    `libcublas.so.13`, `libcublasLt.so.13` and `libnccl.so.2`, as provided by
    the image `nvidia/cuda:13.3.1-runtime-ubuntu24.04` or a CUDA 13.3 toolkit
    install. Built on Ubuntu 24.04, so they need glibc 2.38 or newer. GPU code
    is compiled for compute capabilities 8.6, 8.9 and 12.0 (RTX 30, RTX 40 and
    RTX 50 series).
- The manifest lists each platform's exact needs (`needs.glibc_min`,
  `needs.libraries`).

### The manifest

`potluck-llama-manifest.json`, schema `potluck-llama-release/1`:

| Field | Meaning |
|---|---|
| `tag` | The release tag |
| `llama_build` | Full fork commit the programs were built from. This is the value of `builds.llama` in a Potluck `release.json` |
| `build_number` | Commit count, as printed by `--version` |
| `rpc_protocol` | The ggml RPC protocol version (`RPC_PROTO_*` in `ggml/include/ggml-rpc.h`); host and workers of one load must agree |
| `fork.repository`, `fork.commit` | Where the source is |
| `upstream.repository`, `upstream.base_commit` | The upstream llama.cpp commit the fork's own commits sit on |
| `source_date_epoch` | Commit time, used as the timestamp of every archive entry |
| `workflow_run` | The run that built the release |
| `platforms.<name>` | `os`, `arch`, `backend` (the name is `<os>-<arch>-<backend>`), `archive` (name, sha256, bytes), `files` (sha256 and bytes of `llama-server` and `rpc-server`), `needs`, `cmake_flags`, `toolchain` (image digest, apt snapshot, compiler and SDK versions), `version_output` |

## How Potluck consumes a release

Potluck serves one program folder per Potluck release, described by its own
`release.json`. To use a llama release there:

1. download the release assets and verify them (below);
2. unpack each platform archive into `programs/<potluck release>/`;
3. set `builds.llama` in that `release.json` to the manifest's `llama_build`;
4. install the release's `llama-server` as Potluck's own host program, so the
   host and the workers run the same build.

Potluck itself re-checks every file against the folder's `SHA256SUMS` when it
starts.

## How to cut a release

Releases are cut from the `potluck` branch, which carries the fork's patches.
Tag the commit to release and push the tag:

```sh
git tag -a potluck-llama-v0.1.0 -m "Potluck llama v0.1.0" <commit>
git push origin potluck-llama-v0.1.0
```

The workflow builds both platforms in parallel (expect about an hour, most of
it the CUDA build), assembles the assets, creates the release as a draft, downloads
the uploaded assets again and verifies them, and only then publishes it. A tag
is released once; to rebuild, cut a new tag.

A dry run builds and assembles without publishing: start the workflow by hand
(Actions, "Potluck llama release", "Run workflow") on any branch. The assets
are kept as the run's `release` artifact for seven days.

The same builds run locally with Docker (no GPU needed):

```sh
JOBS=8 .github/potluck/build.sh linux-amd64-vulkan dist
JOBS=8 .github/potluck/build.sh linux-amd64-cuda dist
```

Toolchain versions are pinned in `.github/potluck/toolchain.env`: build images
by digest, Ubuntu packages by exact version from a fixed Ubuntu archive
snapshot. Rebuilding the same commit gives the same programs in behaviour, but
not byte for byte: the compilers embed build paths and temporary file names.

## How to verify a download

```sh
gh release download potluck-llama-v0.1.0 -R Jtonna/llama.cpp-for-datacenter-gpu-orchestrator -D rel
cd rel && sha256sum -c SHA256SUMS
```

then, for a full check of the archives and the programs inside them against the
manifest, from a checkout of this repository:

```sh
.github/potluck/verify.sh rel
```

## Licences

llama.cpp is MIT licensed; its licence is in every archive. The CUDA programs
link dynamically against Nvidia's CUDA runtime, cuBLAS and NCCL, which are not
part of the release: the machine or its container image provides them.
