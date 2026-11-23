# Fork changes

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). For general llama.cpp use, issues and pull requests, go upstream.

| Problem in stock llama.cpp | What the fork changes | Measured effect |
|---|---|---|
| The host uploads model data to its RPC workers one after another | Uploads to all workers at the same time | 1.75x to 1.87x faster loads with two workers, 2.1x to 2.35x with three for models of 12B and up |
| Every load sends all model data across the host's network link; nothing on the workers is reused | Workers load their share from their own copy of the model file | 26.5 GB on six workers: 62.5 s cold, 5.0 s reload; 3 GB on one worker: 26.6 s cold, 1.8 s reload |
| A worker could be pointed at a wrong or stale file, a path outside its folder, or hang a load; a misconfigured worker silently fell back to the slow path | File identity check, path confinement, bounded waits, required mode, progress lines | A misconfigured worker fails the load before any data is sent, a stuck one after a bounded wait, each with one greppable log line naming the cause |

The third row describes the safety work on the fork's own direct-load feature: stock llama.cpp has no direct load, so these are faults that feature would have had without it.

## Why this fork exists

The fork is the llama.cpp build used by Potluck, the scheduler in [Jtonna/datacenter-gpu-orchestrator](https://github.com/Jtonna/datacenter-gpu-orchestrator). Potluck pools mixed machines, such as game consoles, laptops and desktops, into one GPU cluster: one host runs `llama-server` and the other machines run `rpc-server` as workers. Every machine in a load runs programs built from this fork.

Upstream's RPC client sends tensor data to its workers one after another at model load time, and every byte of the model passes through the host. On a cluster of several workers behind one host, model loading is therefore limited by a single connection at a time and by the host's network link. The fork changes how model data reaches the workers. It does not change what is computed.

## Upstream base and the fork's commits

The fork's commits sit on upstream commit [`0a76757469ea94ad5f88d14e18cc7370b9931f99`](https://github.com/ggml-org/llama.cpp/commit/0a76757469ea94ad5f88d14e18cc7370b9931f99) (RPC protocol 10.0.0). To list everything the fork adds:

```sh
git log 0a76757469ea..ps5-direct-pull
```

Besides the eight commits below, that list contains only commits that change CI and documentation: three of 2026-11-22 replace upstream's CI workflows with this fork's release workflow and add [docs/potluck-builds.md](docs/potluck-builds.md), and one adds this file and the README notice. The eight commits that change the programs are:

| Commit | Date | Purpose |
|---|---|---|
| `6cff7a968` | 2026-11-18 | Parallel tensor upload to RPC workers (mmap path) |
| `74ab8e0d0` | 2026-11-20 | Loader fix: every non-mmap read is positioned under a mutex (shared file handle race) |
| `a01665d30` | 2026-11-20 | Loader fix: async upload map lookup under the load mutex (mmap path race) |
| `4b46c3c8c` | 2026-11-20 | RPC workers load tensor data from their own copy of the model file |
| `cb26f3b4d` | 2026-11-20 | Direct file load: file identity check, path confinement, bounded waits, progress log |
| `4c2f23594` | 2026-11-20 | `llama-server` logs model load progress; greppable file-load timeout token |
| `fb295a157` | 2026-11-20 | Direct file load: required mode, announced load totals, finer progress |
| `03c9c2820` | 2026-11-20 | Optional load digest for direct file load; sanitised client strings in worker logs |

Together they touch seven files: `src/llama-model-loader.cpp`, `src/llama-model-loader.h`, `src/llama-model.cpp`, `ggml/include/ggml-rpc.h`, `ggml/src/ggml-rpc/ggml-rpc.cpp`, `tools/rpc/rpc-server.cpp` and `tools/server/server-context.cpp`.

## 1. Parallel tensor upload (`6cff7a968`)

**Fixes:** stock llama.cpp uploads to one RPC worker at a time, so with several workers only one network connection carries data at a time and the others wait.

**Effect:** a 24B model on three workers loads in 57.2 s instead of 129.7 s (2.27x); see the results below.

**What it does.** The model loader starts one thread per (context, buffers) pair instead of loading them one after another, joins them all, then synchronises every RPC backend once. On the mmap path, each tensor in an RPC buffer is sent with an asynchronous set on that buffer's own backend instead of a blocking one, so every worker receives its share at the same time. A cancel flag lets the other threads stop when one fails, and the first error is rethrown.

**Files.** `src/llama-model-loader.cpp`, `src/llama-model-loader.h`, `src/llama-model.cpp`.

**Does not touch.** Anything under `ggml/src/ggml-rpc/`, `ggml-rpc.h`, the RPC messages, commands or protocol version. `--tensor-split` still decides which worker holds which layers, and inference after load is the same as stock. The change only affects how the bytes get to the workers, not which bytes.

**Limits.** The asynchronous upload is used only on the mmap path; without mmap, and when `check_tensors` is on, uploads stay synchronous as upstream. Non-RPC backends that use upstream's pinned host staging keep that path.

**Compatibility.** Works with stock `rpc-server` of the same protocol version. No flags or environment variables.

**History.** An earlier version of this patch was written against upstream `330f0189c` (protocol 8) and is kept on the `ps5-parallel-upload` branch. Rebased onto `0a76757`, the patch became smaller because upstream had meanwhile added a server-advertised hash-cache flag ([`cf92c132`](https://github.com/ggml-org/llama.cpp/commit/cf92c132)), which made the earlier version's own workaround unnecessary.

## 2. Loader thread-safety fixes (`74ab8e0d0`, `a01665d30`)

**Fixes:** two data races that change 1's parallel load introduced; one of them silently loaded wrong weights, with no error, on a host that loads without mmap.

**Effect:** in an off-cluster reproduction, output was wrong on every load before the fix and matched the reference on every load after it.

Both races appear only with the parallel load of change 1.

**Shared file position without mmap (`74ab8e0d0`).** Without mmap, all load threads read through the same file handle, and seek plus read was not atomic, so a thread could read bytes of another tensor: wrong weights, with no error. mmap is off whenever a local device does not support it, which the Vulkan backend reports for integrated GPUs, so this affects a host that has its own integrated GPU. Every read now seeks to its own absolute offset and reads while holding a file mutex. The commit records an off-cluster reproduction (integrated GPU plus two RPC servers): wrong output on every load before the fix, the reference output on every load after it.

**Unlocked map lookup on the mmap path (`a01665d30`).** Each thread looked up the shared async upload map without the lock while sibling threads could still be inserting into it. The lookup now takes the load mutex and copies out only the backend pointer; the upload itself still runs without the lock. This is the path of a CPU-only host with RPC workers.

**Files.** `src/llama-model-loader.cpp`, `src/llama-model-loader.h`.

**Does not touch.** The RPC backend or protocol. No behaviour change apart from removing the races. No flags or environment variables.

## 3. Direct file load on RPC workers (`4b46c3c8c`, `cb26f3b4d`, `4c2f23594`, `fb295a157`, `03c9c2820`)

**Fixes:** with stock llama.cpp every load sends every byte of the model from the host across its network link, even when the workers already hold the same file, and nothing on the workers is reused from one load to the next. The safety work (identity check, path confinement, bounded waits, required mode) closes the faults this feature would otherwise have: loading from a wrong or stale file, reading outside the model folder, hanging a load, and silently falling back to the slow path.

**Effect:** a 26.5 GB model loads onto six workers in 62.5 s cold and 5.0 s on reload; see the results below.

**What it does.** A worker that has its own copy of the model file (for example on a shared network mount) reads its share of the tensor data from that file instead of receiving it from the host. The host sends a file reference per tensor rather than the bytes, and skips its whole-file mmap prefetch so the model is not read through the host. If a worker cannot or may not load from its own copy, the host uploads that worker's data over the socket as before, unless required mode is set.

**Worker side.**

- `rpc-server --model-dir DIR` (short form `-m`) turns the feature on. Without it the worker behaves as stock.
- Path confinement: the requested path is never opened as given. Only its trailing components are looked up below `--model-dir`, longest first. Empty paths, NUL bytes and empty, `.` or `..` components are refused, and symlinks are resolved before the containment check. A refusal answers with a failure and keeps the connection, so the host can fall back.
- File identity: host and worker both compute a digest over the file size, the GGUF header, metadata and tensor-info region, and 64 blocks of 64 KiB at offsets derived from the file size only. This reads the head of the file plus 4 MiB, not the whole file. The worker serves tensor data only from files that passed this check on the same connection, through the same open stream that was hashed. Corruption outside the hashed ranges is not detected.
- Progress: the worker logs `[file_load] progress bytes=N total=M tensors=N tensors_total=K elapsed_ms=N` at most every 64 MiB or 250 ms, and one `[file_load] done ...` line with the same fields on the first graph computation after loading or on disconnect. The total fields are left out when the host did not announce totals.
- `GGML_RPC_FILE_LOAD_DIGEST=1` in the worker's environment makes it hash exactly the bytes it hands to the backend and append `digest=<16 hex>` to the `[file_load] done` line. The digest does not depend on the order in which tensors arrive but does depend on every byte read. Without the variable there is no hashing and no field.
- Client-supplied strings (request paths, tensor names) are printed in worker logs only as bounded, printable ASCII.

**Host side.**

- `LLAMA_RPC_FILE_LOAD`: unset or `1` tries direct load and falls back to socket upload per worker; `0` disables the feature; `required` makes every worker that receives tensors load its share from its own copy, or the load fails.
- In required mode the load fails cleanly (`llama-server` exits 1) without uploading anything over the socket, after one log line per worker and file starting with `rpc_file_load_required: endpoint=H:P file=PATH reason=R`, where `R` is `no_capability` (stock or first-version worker, or no `--model-dir`), `refused` (file not found or unreadable on the worker, or a read failed during the load) or `identity_mismatch` (different size or sampled content).
- The host computes how many tensors and bytes each worker will read from each file and announces them, which gives the worker's progress lines their totals.
- `GGML_RPC_FILE_LOAD_TIMEOUT` (seconds, default 60): the identity check and each file-load request abort the load if the worker has not answered within this time plus 1 s per 8 MiB requested. The abort message starts with `rpc_file_load_timeout: endpoint=... command=... timeout_s=...`. There is no fallback after a timeout because the connection is out of step. Other RPC calls have no new timeout.
- `llama-server` logs `srv  load_progress: fraction=0.42` at most every 250 ms during a model load, with the final `1.00` logged exactly once. Its return value and state reporting are unchanged.

**Wire format.** Two new commands, `RPC_CMD_SET_TENSOR_FILE` and `RPC_CMD_FILE_IDENTITY`, and new bits in the existing HELLO response flags:

| Bit | Meaning |
|---|---|
| `1 << 7` | First version of the extension (size check only). Current hosts do not use it. |
| `1 << 6` | Current extension: file identity check plus file load. Set by workers started with `--model-dir`. |
| `1 << 5` | `RPC_DISPATCHER_FLAGS_FILE_LOAD_TOTALS`: the worker accepts announced totals and returns a reason code. Advertised together with `1 << 6`. |

Existing commands keep their numbers and message layouts, and the protocol version stays 10.0.0, as upstream.

**Compatibility with stock peers.** A stock worker never sets these bits, so a fork host uploads to it over the socket as upstream does (in required mode the load fails with `no_capability`). A stock host ignores the bits and never sends the new commands, so a fork worker serves it as stock. Hosts and workers of the first version of the extension never use file load with current peers; the second version (`1 << 6` only) interoperates with the current one without totals.

**Files.** `ggml/include/ggml-rpc.h`, `ggml/src/ggml-rpc/ggml-rpc.cpp`, `tools/rpc/rpc-server.cpp`, `src/llama-model-loader.cpp`, `src/llama-model-loader.h`, `src/llama-model.cpp`, `tools/server/server-context.cpp` (load progress line only).

**Does not touch.** Tensor allocation, `--tensor-split`, compute and inference, upstream's existing RPC commands, and the protocol version.

## Measured results

### Parallel upload (change 1), November 2026

Cold model load time, stock upstream `0a76757` against the same commit with change 1, both on a host desktop with a 2.5 GbE wired link to a switch and up to three game-console workers with 1 GbE each and an RDNA 2 GPU pinned at 2230 MHz. The patch was validated against this cluster on 2026-11-19.

| Profile | Stock | Patched | Speed-up |
|---|---|---|---|
| tinyllama (1 worker) | 6.9 s | 6.1 s | noise |
| qwen4b (1 worker) | 27.2 s | 27.3 s | 1.00× |
| gemma12b (1 worker) | 62.3 s | 62.4 s | 1.00× |
| qwen4b (2 workers) | 26.2 s | 15.0 s | 1.75× |
| gemma12b (2 workers) | 62.6 s | 33.6 s | 1.87× |
| devstral24b (2 workers) | 130.7 s | 70.9 s | 1.84× |
| qwen27b (2 workers) | 159.9 s | 87.7 s | 1.82× |
| qwen4b (3 workers) | 28.0 s | 15.5 s | 1.81× |
| gemma12b (3 workers) | 62.4-66.4 s | 28.3-32.2 s | ~2.1× |
| devstral24b (3 workers) | 129.7 s | 57.2 s | 2.27× |
| deepseek-v2-lite (3 workers) | 96.5 s | 41.1 s | 2.35× |
| qwen30b-a3b (3 workers) | 178.6 s | 82.7 s | 2.16× |
| qwen27b (3 workers) | 157.7 s | 71.9 s | 2.19× |
| llama33-70b-q2 (3 workers) | 244.8 s | 114.6 s | 2.14× |
| kimi-dev-72b-iq2s (3 workers) | 252.8 s | 115.3 s | 2.19× |
| qwen25-72b-q2 (3 workers) | 273.2 s | 128.9 s | 2.12× |

How to read it: with one worker there is nothing to parallelise, so there is no gain. With two and three workers every worker receives at the same time, so the upload part of the load shrinks roughly by the number of workers until the host's network link is the limit; with three workers that link carried about 290 MB/s in total on 2.5 GbE.

During a patched three-worker load each worker received about 97 MB/s, about 290 MB/s in total; the host's 2.5 GbE link was the limit. Decode speed was the same within noise (1-3%). At temperature 0, fixed seed 42 and the same prompt, the sha256 of the whole response matched stock on all 16 profiles.

### Direct file load (change 3), 2026-11-22

Measured on a cluster of six game consoles used as GPU workers behind one server, on a wired link, each worker loading from its own copy of the model file:

| Model | Workers | Cold load | Reload from the worker's file cache |
|---|---|---|---|
| 3 GB model | 1 | 26.6 s | 1.8 s |
| 26.5 GB model | 6 | 62.5 s | 5.0 s |
| 30B-class model | 2 | about 85 s | - |

How to read it: the reload column shows what direct load makes possible, because each worker reads from its own file cache and no model data crosses the network.

No same-day comparison against the socket-upload path exists yet, so no speed-up factor is claimed for direct load. The reasoning that needs no measurement: with upload, every byte of the model crosses the host's network link on every load; with direct load, none does, and a repeat load reads from the worker's own file cache.

## Branches

| Branch | What it is |
|---|---|
| `ps5-direct-pull` | Default branch and production line: all eight changes, the release workflow and this documentation. Build from this. |
| `master` | Upstream plus older documentation commits. No code changes. |
| `ps5-parallel-upload-head` | Earlier stage: change 1 only, on `0a76757`. Kept for reference. |
| `ps5-loader-fixes` | Earlier stage: changes 1 and 2. Kept for reference. |
| `ps5-parallel-upload` | Earliest stage: the first version of change 1 on the older upstream base `330f0189c` (protocol 8). Kept for reference. |

## Builds and releases

Release builds of `llama-server` and `rpc-server` are made by this repository's release workflow from `potluck-llama-v*` tags. See [docs/potluck-builds.md](docs/potluck-builds.md) for what a release contains, how to cut one and how to verify a download.

## Licence

MIT, unchanged from upstream. The [LICENSE](LICENSE) file is kept as upstream ships it.

## Known open questions

- The file identity check samples the model file and does not prove that every tensor arrived intact on every device. Verifying loaded weights (per-tensor digests) is an open research question, tracked in [issue 1](https://github.com/Jtonna/llama.cpp-for-datacenter-gpu-orchestrator/issues/1).
