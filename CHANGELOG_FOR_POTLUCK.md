# Changelog for Potluck

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), made for one user: Potluck. For general llama.cpp use, issues and pull requests, go upstream.

## Potluck and this build

[Potluck](https://github.com/Jtonna/datacenter-gpu-orchestrator) pools mixed Linux machines, such as game consoles, laptops and desktops, into one cluster that runs large language models behind an OpenAI-compatible API. A model too large for one machine is split across several: one host program, `llama-server`, and one `rpc-server` worker on each other machine.

How Potluck uses this build, as its code and its [agent contract](https://github.com/Jtonna/datacenter-gpu-orchestrator/blob/main/docs/agent-contract.md) show:

- **Same build everywhere.** Host and workers must come from the same fork commit. Potluck checks at start that its own `llama-server` was built from the commit its release names, checks every program file it serves to the workers, and refuses to start any load when a check fails (agent contract, section 5.1).
- **Workers.** Potluck's agent starts `rpc-server` on each worker. When the pool setting `worker_model_dir` is on, it adds `--model-dir` pointing at the node's model folder: the model share mounted read-only over NFS, or a model folder given at install (section 8).
- **Host.** For a load with workers, Potluck sets `LLAMA_RPC_FILE_LOAD=required` on the host `llama-server` when `worker_model_dir` is on, and `LLAMA_RPC_FILE_LOAD=0` (the host sends the weights) when it is off (section 11.1).
- **Logs.** Potluck reads load progress from the host's `load_progress: fraction=` line and the workers' `[file_load]` lines, and reports a failed load from the `rpc_file_load_required:` and `rpc_file_load_timeout:` lines (section 6.3).

## At a glance

| Problem in stock llama.cpp | What the fork changes | Measured effect |
|---|---|---|
| The host uploads model data to its workers one after another | Uploads to all workers at the same time | 1.75x to 1.87x faster loads with two workers, 2.1x to 2.35x with three for models of 12B and up |
| Every load sends the whole model from the host program to the workers over its RPC connections | Each worker reads its share from its own mount (or copy) of the model file | 26.5 GB on six workers: 62.5 s cold, 5.0 s reload; 3 GB on one worker: 26.6 s cold, 1.8 s reload |
| (New with direct load) a worker could read a wrong or stale file, a path outside its folder, or hang a load; a misconfigured worker would silently fall back to the slow path | Identity check, path confinement, bounded waits, required mode, progress lines | A misconfigured worker fails the load before any data is sent, a stuck one after a bounded wait, each with one greppable log line naming the cause |

This file has two parts: the change log (what and why, oldest change first), then a technical reference (flags, variables, log lines, wire compatibility).

# Part 1: Change log

## 1. Parallel upload to workers (2026-11-18, `6cff7a968`)

- **What Potluck needed:** models split across two or three consoles loaded slowly, and adding a worker did not make loading faster.
- **What stock llama.cpp did:** the host sent each worker its share in turn, so only one network connection carried data at a time while the other workers waited.
- **What the fork changed:** the host loads all workers' shares at the same time, each over its own connection. Which worker holds which layers, and what is computed, is unchanged.
- **How Potluck uses it:** always; there is no setting. It is how the host sends the weights when `worker_model_dir` is off.
- **Measured effect:** a 24B model on three workers loads in 57.2 s instead of 129.7 s (2.27x). Answers at temperature 0 were byte-identical to stock on all 16 tested profiles.

## 2. Two loader thread-safety fixes (2026-11-20, `74ab8e0d0`, `a01665d30`)

- **What Potluck needed:** correct weights on every load.
- **What went wrong:** change 1's parallel load introduced two data races. Without mmap, threads shared one file position, so a thread could read another tensor's bytes: wrong weights, no error. With mmap, threads read a shared map while another thread could still be writing it.
- **What the fork changed:** every read without mmap seeks and reads under a lock; the map lookup takes the load lock.
- **How Potluck uses it:** every load runs through the parallel loader, so both fixes apply to every load.
- **Measured effect:** in an off-cluster reproduction the output was wrong on every load before the fix and matched the reference on every load after it.

## 3. Workers read their share from the model file themselves (2026-11-20, `4b46c3c8c`)

- **What Potluck needed:** faster loads and reloads without giving every console its own copy of every model. The consoles' storage is not to be written with models, so Potluck keeps each model file once, on a file server, and every worker mounts that share read-only.
- **What stock llama.cpp did:** the host program read the whole file and sent every byte to the workers over its RPC connections on every load.
- **What the fork changed:** a worker started with `--model-dir` can open the same file at a path inside that folder and read its share itself; the host sends a file reference per tensor instead of the bytes. The folder can be a network mount (the tested case) or a local copy (possible, not measured). On a cold load each worker pulls its share from the file server over the network, all workers in parallel, without passing through the host's upload path. On a repeat load the worker reads from its operating system's file cache of that mount, which is why a reload takes seconds.
- **How Potluck uses it:** when `worker_model_dir` is on, workers get `--model-dir` on the mounted share.
- **Measured effect:** a 26.5 GB model on six workers: 62.5 s cold, 5.0 s reload. See the results below.

## 4. Making direct load safe and visible (2026-11-20, `cb26f3b4d`, `4c2f23594`, `fb295a157`, `03c9c2820`)

- **What Potluck needed:** a load that either uses the right file or fails at once with a clear reason, and progress it can show while a large model loads.
- **What could go wrong without this:** the first version compared file sizes only, so any same-size file was accepted; a request could name a path outside the model folder; a worker that stopped answering could hang the load; a worker that could not read the file made the load fall back to the slow upload without anyone noticing; and a long load showed no progress.
- **What the fork changed:** a file identity check before any tensor data is read; paths confined to `--model-dir`; bounded waits; a required mode that fails the load with one named reason per worker; progress lines on the host and on every worker with announced totals; an optional digest of the bytes a worker loaded; bounded, printable log output of client-supplied strings.
- **How Potluck uses it:** it sets required mode when workers read the file, reads the progress lines for its load progress, and turns the two failure lines into a named load failure.
- **Measured effect:** a misconfigured worker fails the load before any tensor data is sent; a stuck one after a bounded wait (60 s plus 1 s per 8 MiB by default).

## Measured results

These are measurements on the fork's authors' own cluster and network, not general benchmarks. Other hardware or networks will give different numbers.

### Parallel upload (change 1)

Test setup: one server with a 2.5 GbE wired link to a switch and up to three game consoles as GPU workers with 1 GbE each; stock upstream `0a76757` against the same commit with change 1; November 2026 (validated on 2026-11-19).

| Profile | Stock | Patched | Speed-up |
|---|---|---|---|
| tinyllama (1 worker) | 6.9 s | 6.1 s | noise |
| qwen4b (1 worker) | 27.2 s | 27.3 s | 1.00x |
| gemma12b (1 worker) | 62.3 s | 62.4 s | 1.00x |
| qwen4b (2 workers) | 26.2 s | 15.0 s | 1.75x |
| gemma12b (2 workers) | 62.6 s | 33.6 s | 1.87x |
| devstral24b (2 workers) | 130.7 s | 70.9 s | 1.84x |
| qwen27b (2 workers) | 159.9 s | 87.7 s | 1.82x |
| qwen4b (3 workers) | 28.0 s | 15.5 s | 1.81x |
| gemma12b (3 workers) | 62.4-66.4 s | 28.3-32.2 s | ~2.1x |
| devstral24b (3 workers) | 129.7 s | 57.2 s | 2.27x |
| deepseek-v2-lite (3 workers) | 96.5 s | 41.1 s | 2.35x |
| qwen30b-a3b (3 workers) | 178.6 s | 82.7 s | 2.16x |
| qwen27b (3 workers) | 157.7 s | 71.9 s | 2.19x |
| llama33-70b-q2 (3 workers) | 244.8 s | 114.6 s | 2.14x |
| kimi-dev-72b-iq2s (3 workers) | 252.8 s | 115.3 s | 2.19x |
| qwen25-72b-q2 (3 workers) | 273.2 s | 128.9 s | 2.12x |

How to read it: with one worker there is nothing to parallelise, so there is no gain. With two and three workers every worker receives at the same time, so the upload part of the load shrinks roughly by the number of workers until the server's network link is the limit; with three workers each received about 97 MB/s, about 290 MB/s in total on 2.5 GbE. Generation speed was the same within noise (1-3%). At temperature 0, fixed seed 42 and the same prompt, the sha256 of the whole response matched stock on all 16 profiles.

### Direct file load (change 3)

Test setup: 2026-11-22, six game consoles as GPU workers with 1 GbE each, one server on a 5 GbE wired link, a wired switch. The server's link was upgraded from 2.5 GbE between the two tests, so the two tables are not a like-for-like comparison. The model file is stored once on the server and shared read-only to the workers over the network, by choice, so that no worker has to own a copy; each worker reads its share from that mount. Results on other hardware, networks or with local copies will differ.

| Model | Workers | Cold load | Reload |
|---|---|---|---|
| 3 GB model | 1 | 26.6 s | 1.8 s |
| 26.5 GB model | 6 | 62.5 s | 5.0 s |
| 30B-class model | 2 | about 85 s | - |

How to read it: a cold load is each worker pulling its share from the file server over the network, all at once; a reload is each worker reading from its own file cache of the mount, so no model data crosses the network. The cold load is limited by the network between the file server and the workers (about 424 MB/s in total here), not by the host program.

No same-day comparison against the socket-upload path exists yet, so no speed-up factor is claimed for direct load. The reasoning that needs no measurement: with socket upload the host program sends every byte of the model to the workers on every load; with direct load it sends none, and a repeat load is served from each worker's file cache.

# Part 2: Technical reference

## Upstream base and commits

The fork's commits sit on upstream commit [`0a76757469ea94ad5f88d14e18cc7370b9931f99`](https://github.com/ggml-org/llama.cpp/commit/0a76757469ea94ad5f88d14e18cc7370b9931f99) (RPC protocol 10.0.0). To list everything the fork adds:

```sh
git log 0a76757469ea..potluck
```

Besides the eight commits below, that list contains only commits that change CI and documentation: three of 2026-11-22 replace upstream's CI workflows with this fork's release workflow and add [docs/potluck-builds.md](docs/potluck-builds.md), and the others add and update this file and the README notice.

| Commit | Date | Purpose |
|---|---|---|
| `6cff7a968` | 2026-11-18 | Parallel tensor upload to RPC workers (mmap path) |
| `74ab8e0d0` | 2026-11-20 | Loader fix: every non-mmap read is positioned under a mutex (shared file handle race) |
| `a01665d30` | 2026-11-20 | Loader fix: async upload map lookup under the load mutex (mmap path race) |
| `4b46c3c8c` | 2026-11-20 | RPC workers load tensor data from the model file themselves |
| `cb26f3b4d` | 2026-11-20 | Direct file load: file identity check, path confinement, bounded waits, progress log |
| `4c2f23594` | 2026-11-20 | `llama-server` logs model load progress; greppable file-load timeout token |
| `fb295a157` | 2026-11-20 | Direct file load: required mode, announced load totals, finer progress |
| `03c9c2820` | 2026-11-20 | Optional load digest for direct file load; sanitised client strings in worker logs |

Together they touch seven files: `src/llama-model-loader.cpp`, `src/llama-model-loader.h`, `src/llama-model.cpp`, `ggml/include/ggml-rpc.h`, `ggml/src/ggml-rpc/ggml-rpc.cpp`, `tools/rpc/rpc-server.cpp` and `tools/server/server-context.cpp`.

## Parallel upload

- **Mechanism.** The model loader starts one thread per (context, buffers) pair, joins them all, then synchronises every RPC backend once. On the mmap path each tensor in an RPC buffer is sent with an asynchronous set on that buffer's own backend. A cancel flag stops the other threads when one fails, and the first error is rethrown.
- **Files.** `src/llama-model-loader.cpp`, `src/llama-model-loader.h`, `src/llama-model.cpp`.
- **Does not touch.** Anything under `ggml/src/ggml-rpc/`, `ggml-rpc.h`, the RPC messages, commands or protocol version, or `--tensor-split`.
- **Limits.** The asynchronous upload is used only on the mmap path; without mmap, and with `check_tensors`, uploads stay synchronous as upstream. Non-RPC backends that use upstream's pinned host staging keep that path.
- **Compatibility.** Works with stock `rpc-server` of the same protocol version. No flags or environment variables.
- **History.** A first version against upstream `330f0189c` (protocol 8) is kept on the `ps5-parallel-upload` branch. Rebased onto `0a76757` the patch became smaller, because upstream had added a server-advertised hash-cache flag ([`cf92c132`](https://github.com/ggml-org/llama.cpp/commit/cf92c132)).

## Loader fixes

- **Non-mmap race (`74ab8e0d0`).** Every read (host buffer, async chunk, staging buffer, range load) seeks to its own absolute offset and reads while holding a file mutex. mmap is off whenever a local device does not support it, which the Vulkan backend reports for integrated GPUs, so this affects a host with its own integrated GPU.
- **mmap race (`a01665d30`).** The lookup in the shared async upload map takes the load mutex and copies out only the backend pointer; the upload itself runs without the lock. This is the path of a CPU-only host with RPC workers.
- **Files.** `src/llama-model-loader.cpp`, `src/llama-model-loader.h`. No flags, no protocol change.

## Direct file load

**Worker (`rpc-server`).**

- `--model-dir DIR` (short `-m`) turns the feature on. Without it the worker behaves as stock.
- Path confinement: the requested path is never opened as given. Only its trailing components are looked up below `--model-dir`, longest first. Empty paths, NUL bytes and empty, `.` or `..` components are refused, and symlinks are resolved before the containment check. A refusal keeps the connection, so the host can fall back.
- File identity: host and worker both compute a digest over the file size, the GGUF header, metadata and tensor-info region, and 64 blocks of 64 KiB at offsets derived from the file size only (the head of the file plus 4 MiB, not the whole file). The worker serves tensor data only from files that passed this check on the same connection, through the same open stream that was hashed. Corruption outside the hashed ranges is not detected.
- Progress: `[file_load] progress bytes=N total=M tensors=N tensors_total=K elapsed_ms=N` at most every 64 MiB or 250 ms, and one `[file_load] done ...` line with the same fields on the first graph computation after loading or on disconnect. The total fields are left out when the host announced none.
- `GGML_RPC_FILE_LOAD_DIGEST=1` in the worker's environment hashes exactly the bytes handed to the backend and appends `digest=<16 hex>` to the `done` line. The digest does not depend on tensor arrival order but does depend on every byte read.
- Client-supplied strings (request paths, tensor names) appear in worker logs only as bounded, printable ASCII.

**Host (`llama-server` and the model loader).**

- `LLAMA_RPC_FILE_LOAD`: unset or `1` tries direct load and falls back to socket upload per worker; `0` disables the feature; `required` makes every worker that receives tensors read its share itself, or the load fails.
- Required mode fails the load cleanly (`llama-server` exits 1) without uploading anything over the socket, after one line per worker and file starting with `rpc_file_load_required: endpoint=H:P file=PATH reason=R`. `R` is `no_capability` (stock or first-version worker, or no `--model-dir`), `refused` (file not found or unreadable on the worker, or a read failed during the load) or `identity_mismatch` (different size or sampled content). Capability is checked before any buffer is allocated, identity before any tensor data is sent.
- The host computes how many tensors and bytes each worker will read from each file and announces them, which gives the worker's progress lines their totals.
- `GGML_RPC_FILE_LOAD_TIMEOUT` (seconds, default 60): the identity check and each file-load request abort the load if the worker has not answered within this time plus 1 s per 8 MiB requested. The message starts with `rpc_file_load_timeout: endpoint=... command=... timeout_s=...`. There is no fallback after a timeout because the connection is out of step. Other RPC calls have no new timeout.
- When any worker can load the file itself, the host skips its whole-file mmap prefetch, so the model is not read through the host.
- `llama-server` logs `srv  load_progress: fraction=0.42` at most every 250 ms during a model load, with the final `1.00` exactly once. Its return value and state reporting are unchanged.

**Wire format.** Two new commands, `RPC_CMD_SET_TENSOR_FILE` and `RPC_CMD_FILE_IDENTITY`, and new bits in the existing HELLO response flags:

| Bit | Meaning |
|---|---|
| `1 << 7` | First version of the extension (size check only). Current hosts do not use it. |
| `1 << 6` | Current extension: file identity check plus file load. Set by workers started with `--model-dir`. |
| `1 << 5` | `RPC_DISPATCHER_FLAGS_FILE_LOAD_TOTALS`: the worker accepts announced totals and returns a reason code. Advertised together with `1 << 6`. |

Existing commands keep their numbers and message layouts, and the protocol version stays 10.0.0, as upstream.

**Compatibility with stock peers.** A stock worker never sets these bits, so a fork host uploads to it over the socket as upstream does (in required mode the load fails with `no_capability`). A stock host ignores the bits and never sends the new commands, so a fork worker serves it as stock. First-version peers never use file load with current peers; second-version peers (`1 << 6` only) interoperate with current ones without totals.

**Files.** `ggml/include/ggml-rpc.h`, `ggml/src/ggml-rpc/ggml-rpc.cpp`, `tools/rpc/rpc-server.cpp`, `src/llama-model-loader.cpp`, `src/llama-model-loader.h`, `src/llama-model.cpp`, `tools/server/server-context.cpp` (load progress line only).

**Does not touch.** Tensor allocation, `--tensor-split`, compute and inference, upstream's existing RPC commands, and the protocol version.

## Branches

| Branch | What it is |
|---|---|
| `potluck` | Default branch and production line: all eight changes, the release workflow and this documentation. Build from this. Called `ps5-direct-pull` until 23 November 2026 (UTC). |
| `master` | Upstream plus older documentation commits. No code changes. |
| `ps5-parallel-upload-head` | Earlier stage: change 1 only, on `0a76757`. Kept for reference. |
| `ps5-loader-fixes` | Earlier stage: changes 1 and 2. Kept for reference. |
| `ps5-parallel-upload` | Earliest stage: the first version of change 1 on the older upstream base `330f0189c` (protocol 8). Kept for reference. |

## Builds and releases

Release builds of `llama-server` and `rpc-server` are made by this repository's release workflow from `potluck-llama-v*` tags. See [docs/potluck-builds.md](docs/potluck-builds.md) for what a release contains, how to cut one, how to verify a download and how Potluck consumes it.

## Licence

MIT, unchanged from upstream. The [LICENSE](LICENSE) file is kept as upstream ships it.

## Known open questions

- The file identity check samples the model file and does not prove that every tensor arrived intact on every device. Verifying loaded weights (per-tensor digests) is an open research question, tracked in [issue 1](https://github.com/Jtonna/llama.cpp-for-datacenter-gpu-orchestrator/issues/1).
