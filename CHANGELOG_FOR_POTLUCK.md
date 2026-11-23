# Changelog for Potluck

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), made for one user: Potluck. For general llama.cpp use, issues and pull requests, go upstream.

## Potluck and this build

[Potluck](https://github.com/Jtonna/datacenter-gpu-orchestrator) pools mixed Linux machines, such as game consoles, laptops and desktops, into one cluster that runs large language models behind an OpenAI-compatible API. A model too large for one machine is split across several: one host program, `llama-server`, and one `rpc-server` worker on each other machine.

How Potluck uses this build, as its code and its [agent contract](https://github.com/Jtonna/datacenter-gpu-orchestrator/blob/main/docs/agent-contract.md) show:

- **Same build everywhere.** Host and workers must come from the same fork commit. Potluck checks at start that its own `llama-server` reports (`--version`) the commit its release names, hashes every program file of the release against its `SHA256SUMS`, and refuses to serve the release, and so to start any load, when a check fails (agent contract, sections 5.1 and 5.2).
- **Workers.** Potluck's agent starts `rpc-server` on each worker. When the pool setting `worker_model_dir` is on (it is off by default), the agent adds `--model-dir` pointing at the node's model folder, on the consoles the model share mounted read-only over NFS (sections 6.2 and 8).
- **Host.** For a load on agent workers, Potluck sets `LLAMA_RPC_FILE_LOAD=required` on its host `llama-server` when `worker_model_dir` is on, and `LLAMA_RPC_FILE_LOAD=0` (the host sends the weights) when it is off (section 11.1).
- **Logs.** Potluck reads load progress from the host's `load_progress: fraction=` line and from the workers' `[file_load]` lines that carry `bytes=` and `total=`, and reports a failed load from the `rpc_file_load_required:` and `rpc_file_load_timeout:` tokens in the host log (section 6.3).

## At a glance

| Problem in stock llama.cpp | What the fork changes | Effect |
|---|---|---|
| The host loads one device's share after another, so its workers receive their weights one at a time | Loads all devices' shares at the same time, each worker over its own connection | Measured: 1.75x to 1.86x faster loads with two workers, about 2.1x to 2.35x with three for models of 12B and up |
| On every load the host reads the model and sends the workers' shares over its RPC connections (upstream's optional `rpc-server -c` cache can skip large tensors, but stores the model on the worker's disk) | Each worker reads its share from its own mount (or copy) of the model file | Measured: 26.5 GB on six workers, 62.5 s cold, 5.0 s reload; 3 GB on one worker, 26.6 s cold, 1.8 s reload. No same-day comparison with socket upload |
| (New with direct load) the first version accepted any file of the right size, could wait forever on a stalled worker, fell back to the slow path with only a warning, and showed no progress | Identity check on sampled content, stricter path rules, bounded waits, required mode, progress lines | By design, not measured: in required mode a misconfigured worker fails the load before any tensor data is sent, with one greppable line naming the cause; a worker that stops answering makes the host abort after a bounded wait |

This file has two parts: the change log (what and why, oldest change first), then a technical reference (flags, variables, log lines, wire compatibility).

# Part 1: Change log

## 1. Parallel upload to workers (2026-11-18, `6cff7a968`)

- **What Potluck needed:** models split across two or three consoles loaded slowly, and adding a worker did not make loading faster (stock: 27.2 s, 26.2 s and 28.0 s for the same 4B model on one, two and three workers).
- **What stock llama.cpp did:** the loader keeps one context per buffer type, which for RPC means one per worker device, and loaded these contexts one after another. On the default mmap path each tensor for a worker was sent with a blocking call that returned once its bytes were written to that worker's connection. So only one worker received data at a time while the others waited.
- **What the fork changed:** each context is loaded on its own thread, so all workers receive their shares at the same time, each over its own connection. On the mmap path each tensor is queued with an asynchronous send, and every worker is synchronised once after all threads have finished. Which worker holds which layers, and what is computed, is unchanged.
- **How Potluck uses it:** always; there is no setting. With `worker_model_dir` off it is how the host sends the weights; with it on, the same threads send the workers' file-load requests at the same time.
- **Measured effect:** a 24B model on three workers loads in 57.2 s instead of 129.7 s (2.27x). At temperature 0 with a fixed seed, answers were byte-identical to stock on all 16 tested profiles. Both measured on this commit alone, on 2026-11-19, before changes 2 to 4.

## 2. Two loader thread-safety fixes (2026-11-20, `74ab8e0d0`, `a01665d30`)

- **What Potluck needed:** correct weights on every load.
- **What went wrong:** change 1's parallel load introduced two data races; upstream loads one context at a time and has neither. Without mmap, all threads read through one file handle with one shared position, and seek and read were separate steps, so a thread could read another tensor's bytes: wrong weights, no error. With mmap, a thread looked up the shared table of upload backends without the lock while another thread could still be inserting into it: undefined behaviour (a crash, or a missed lookup that falls back to a slower blocking send).
- **What the fork changed:** every read without mmap positions itself and reads under a file lock; the table lookup takes the load lock.
- **How Potluck uses it:** every load runs through the parallel loader, so one of the two fixes is on its path. In the measured setup the host runs on the server's CPU with no GPU layers and loads with mmap, so the mmap fix applies; the other matters for a host that loads without mmap, for example one with an integrated GPU.
- **Measured effect:** not measured on the cluster. The non-mmap race was reproduced off-cluster (a host with an integrated GPU and two RPC workers, a small model) by stalling one thread around its seek: wrong output on every load before the fix (6 of 6) and the reference output on every load after it (6 of 6). The commit for the mmap race records no reproduction; [issue 1](https://github.com/Jtonna/llama.cpp-for-datacenter-gpu-orchestrator/issues/1) reports it was confirmed with ThreadSanitizer.

## 3. Workers read their share from the model file themselves (2026-11-20, `4b46c3c8c`)

- **What Potluck needed:** faster loads and reloads without giving every console its own copy of every model. The consoles' storage is not to be written with models, so Potluck keeps each model file once, on a file server, and every worker mounts that share read-only.
- **What stock llama.cpp did:** on every load the host program read every tensor from its own copy of the file and sent the workers' shares over its RPC connections. Upstream's optional worker cache (`rpc-server -c`) lets a later load skip tensors over 10 MiB that the worker already holds, but the host still reads and hashes them, and the cache writes the model to the worker's disk, which Potluck does not allow.
- **What the fork changed:** a worker started with `--model-dir` can open the same file at a path inside that folder and read its share itself; the host sends a file reference (path, offset, size) per tensor instead of the bytes. The folder can be a network mount (the tested case) or a local copy (possible, not measured). On a cold load each worker reads its share from the file server over the network, all workers at the same time, and none of it passes through the host. The worker reads with ordinary buffered file reads (no mmap, no direct I/O), so its operating system's page cache keeps what it read, and a repeat load is fast while the worker's share is still in that cache.
- **How Potluck uses it:** when `worker_model_dir` is on, workers get `--model-dir` on the mounted share.
- **Measured effect:** a 26.5 GB model on six workers: 62.5 s cold, 5.0 s reload. See the results below.

## 4. Making direct load safe and visible (2026-11-20, `cb26f3b4d`, `4c2f23594`, `fb295a157`, `03c9c2820`)

- **What Potluck needed:** a load that either uses the right file or fails at once with a clear reason, and progress it can show while a large model loads.
- **What could go wrong with the first version (`4b46c3c8c`):** it compared file sizes only, so any same-size file was accepted; it opened the client's path as given when that path resolved inside the model folder (files resolving outside it were already refused); a worker that stopped answering could hang the load; a worker that refused made the load fall back to the slow upload with only a warning; and a long load showed no progress.
- **What the fork changed:** a file identity check before any tensor data is read; the client's path is never opened as given, only looked up inside `--model-dir`; bounded waits; a required mode that fails the load with one named reason per worker and file; progress lines on the host and on every worker, with totals announced by the host; an optional digest of the bytes a worker loaded; client-supplied strings logged by the worker as bounded, printable text.
- **How Potluck uses it:** it sets required mode when workers read the file, reads the progress lines for its load progress, and turns the two failure tokens into a named load failure.
- **Effect (by design, not measured):** in required mode a misconfigured worker fails the load before any tensor data is sent; a worker that stops answering makes the host abort after a bounded wait (60 s plus 1 s per 8 MiB requested, by default).

## Measured results

These are measurements on the fork's authors' own cluster and network, not general benchmarks. Other hardware or networks will give different numbers.

### Parallel upload (change 1)

Test setup: one server with a 2.5 GbE wired link to a switch and up to three game consoles as GPU workers with 1 GbE each; stock upstream `0a76757` against the same commit with change 1 only; validated on 2026-11-19.

| Profile | Stock | Patched | Speed-up |
|---|---|---|---|
| tinyllama (1 worker) | 6.9 s | 6.1 s | noise |
| qwen4b (1 worker) | 27.2 s | 27.3 s | 1.00x |
| gemma12b (1 worker) | 62.3 s | 62.4 s | 1.00x |
| qwen4b (2 workers) | 26.2 s | 15.0 s | 1.75x |
| gemma12b (2 workers) | 62.6 s | 33.6 s | 1.86x |
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

How to read it: with one worker there is nothing to parallelise, so there is no gain. With two and three workers every worker receives at the same time, so the upload part of the load shrinks roughly by the number of workers until the server's network link is the limit; the whole load also includes steps that do not shrink. The original measurement notes (not part of this table) record about 97 MB/s per worker and about 290 MB/s in total with three workers on the 2.5 GbE link. Generation speed was the same within noise (1-3%). At temperature 0, fixed seed 42, prompt caching off and the same prompt, the sha256 of the whole response matched stock on all 16 profiles.

### Direct file load (change 3)

Test setup: 2026-11-22, Potluck with `worker_model_dir` on, six game consoles as GPU workers with 1 GbE each, the host program on the server's CPU with no GPU layers, the server on a 5 GbE wired link, a wired switch. The server's link was upgraded from 2.5 GbE between the two tests, so the two tables are not a like-for-like comparison. The model file is stored once on the server and shared read-only to the workers over NFS, by choice, so that no worker has to own a copy. Cold: the model file had been dropped from the page cache before the load; the 3 GB row is Potluck's own verified cold load, which drops it on every worker and on the server. Results on other hardware, networks or with local copies will differ.

| Model | Workers | Cold load | Reload |
|---|---|---|---|
| 3 GB model | 1 | 26.6 s | 1.8 s |
| 26.5 GB model | 6 | 62.5 s | 5.0 s |
| 30B-class model | 2 | about 85 s (cache state not recorded) | - |

Model sizes are approximate; the result files do not record them. The 30B-class figure comes from a later run of the same day that loaded the model and then held a long conversation.

How to read it: on a cold load each worker reads its share from the file server over the network, all at the same time. 26.5 GB in 62.5 s is about 424 MB/s on average across the six workers, counted over the whole load; which link or disk limited it was not measured. A reload is served from each worker's page cache while its share stays cached beside the loaded model; in the same test series, on placements tight enough that part of the cached file was pushed out, reloads took 10 to 40 s.

No same-day comparison against the socket-upload path exists yet, so no speed-up factor is claimed for direct load. The reasoning that needs no measurement: with socket upload the host program sends every byte of the workers' shares on every load (unless the workers keep upstream's `-c` cache); with direct load it sends none of them.

# Part 2: Technical reference

## Upstream base and commits

The fork's commits sit on upstream commit [`0a76757469ea94ad5f88d14e18cc7370b9931f99`](https://github.com/ggml-org/llama.cpp/commit/0a76757469ea94ad5f88d14e18cc7370b9931f99) (RPC protocol 10.0.0). To list everything the fork adds:

```sh
git log 0a76757469ea..potluck
```

Besides the eight commits below, that list contains only commits that change CI and documentation: three of 2026-11-22 replace upstream's CI workflows with this fork's release workflow and add [docs/potluck-builds.md](docs/potluck-builds.md), and the others add and update the README notice, [FORK_CHANGES.md](FORK_CHANGES.md) and this file.

| Commit | Date | Purpose |
|---|---|---|
| `6cff7a968` | 2026-11-18 | Parallel tensor upload to RPC workers (asynchronous sends on the mmap path) |
| `74ab8e0d0` | 2026-11-20 | Loader fix: every non-mmap read is positioned under a mutex (shared file handle race) |
| `a01665d30` | 2026-11-20 | Loader fix: async upload map lookup under the load mutex (mmap path race) |
| `4b46c3c8c` | 2026-11-20 | RPC workers load tensor data from the model file themselves (first version: size check only) |
| `cb26f3b4d` | 2026-11-20 | Direct file load: file identity check, stricter path rules, bounded waits, progress log |
| `4c2f23594` | 2026-11-20 | `llama-server` logs model load progress; greppable file-load timeout token |
| `fb295a157` | 2026-11-20 | Direct file load: required mode, announced load totals, finer progress |
| `03c9c2820` | 2026-11-20 | Optional load digest for direct file load; sanitised client strings in worker logs |

Together they touch seven files: `src/llama-model-loader.cpp`, `src/llama-model-loader.h`, `src/llama-model.cpp`, `ggml/include/ggml-rpc.h`, `ggml/src/ggml-rpc/ggml-rpc.cpp`, `tools/rpc/rpc-server.cpp` and `tools/server/server-context.cpp`.

## Parallel upload

- **Mechanism.** The model loader starts one thread per context (one context per buffer type: one per worker device and one per local device), joins them all, then synchronises each asynchronous upload backend once. On the mmap path, a tensor in a non-host buffer whose device supports asynchronous sets and events is sent with an asynchronous set on a backend created for that device; this covers RPC workers, and also local GPUs that report these capabilities. Shared loader state is updated under one load lock. If the progress callback cancels the load, or a worker refuses a read in required mode, a flag stops the other threads at their next tensor; an exception from any thread is rethrown after all threads have finished.
- **Files.** `src/llama-model-loader.cpp`, `src/llama-model-loader.h`, `src/llama-model.cpp`.
- **Does not touch.** Anything under `ggml/src/ggml-rpc/`, `ggml-rpc.h`, the RPC messages, commands or protocol version, or `--tensor-split`.
- **Limits.** The asynchronous set is used only on the mmap path and not with `check_tensors`. Without mmap the threads still run at the same time, but each tensor is read under the file lock and sent with a blocking set; upstream's pinned-memory staging for devices that offer host buffers is unchanged. Only the mmap path with RPC workers was measured.
- **Compatibility.** Works with stock `rpc-server` of the same protocol version. No flags or environment variables.
- **History.** A first version against upstream `330f0189c` (protocol 8) is kept on the `ps5-parallel-upload` branch. Rebased onto `0a76757` the patch became smaller, because upstream had added a server-advertised hash-cache flag ([`cf92c132`](https://github.com/ggml-org/llama.cpp/commit/cf92c132)), so the first version's changes to `ggml-rpc.cpp` were no longer needed.

## Loader fixes

- **Non-mmap race (`74ab8e0d0`).** Every read (host buffer, async chunk, staging buffer, range load) seeks to its own absolute offset and reads while holding a file mutex. mmap is off when the load mode turns it off, or in the default mode whenever a local device does not support it, which the Vulkan backend reports for integrated GPUs.
- **mmap race (`a01665d30`).** The lookup in the shared async upload map takes the load mutex and copies out only the backend pointer; the send itself runs without the lock. This is the path of a host that loads with mmap, such as a CPU-only host with RPC workers.
- **Files.** `src/llama-model-loader.cpp`, `src/llama-model-loader.h`. No flags, no protocol change.

## Direct file load

**Worker (`rpc-server`).**

- `--model-dir DIR` (short `-m`) turns the feature on; `rpc-server` refuses to start if DIR is not an existing directory. Without it the worker advertises nothing new and refuses the new commands.
- Path confinement: the requested path is never opened as given. Only its trailing components are looked up below `--model-dir`, longest first. Empty paths, NUL bytes and empty, `.` or `..` components are refused. The folder and every candidate are resolved to canonical paths (symlinks followed) and the candidate must be a regular file inside the folder. A refusal keeps the connection, so the host can fall back.
- File identity: host and worker both compute a 64-bit FNV-1a hash (not cryptographic) over the file size, the region from the start of the file to the start of tensor data (GGUF header, metadata and tensor infos, at most 256 MiB), and 64 blocks of 64 KiB at offsets spread evenly over the file, derived from its size only: the head plus 4 MiB, not the whole file. Sizes must match too. The worker serves tensor data only from files that passed this check on the same connection, through the same open stream that was hashed. Corruption outside the hashed ranges is not detected.
- Reads: ordinary buffered file reads in chunks of up to 16 MiB, through the operating system's page cache, staged through a buffer for non-CPU devices.
- Progress: `[file_load] progress bytes=N total=M tensors=N tensors_total=K elapsed_ms=N`, written when 64 MiB more have been read or 250 ms have passed since the previous line (checked after each chunk), and one `[file_load] done ...` line with the same fields on the first graph computation after loading or on disconnect. The total fields are left out when the host announced none.
- `GGML_RPC_FILE_LOAD_DIGEST=1` in the worker's environment hashes exactly the bytes handed to the backend and appends `digest=<16 hex>` to the `done` line. The digest does not depend on tensor arrival order but does depend on every byte read. The host neither computes nor checks it; it is for comparing loads and workers.
- Client-supplied request paths in the file-load log lines, and the tensor name in the out-of-bounds error, are logged as bounded, printable ASCII (other bytes become `?`).

**Host (`llama-server` and the model loader).**

- `LLAMA_RPC_FILE_LOAD`, read by the model loader: `0` disables the feature; `required` makes every worker that receives tensors read its share itself, or the load fails; any other value, including unset and `1`, tries direct load and falls back to socket upload for a worker and file that fail the check, and for all of a worker's remaining tensors after it refuses a read.
- Required mode fails the load cleanly (`llama-server` exits 1) without sending any tensor data over the socket. Up-front failures log one line per worker and file starting with `rpc_file_load_required: endpoint=H:P file=PATH reason=R`; a read refused during the load logs one such line with `reason=refused`. `R` is `no_capability` (stock or first-version worker, or no `--model-dir`), `refused` (file not found inside the worker's `--model-dir`, unreadable, or a read failed during the load) or `identity_mismatch` (different size or sampled content). Capability is checked before any buffer is allocated, identity before any tensor data is sent. Required mode cannot be combined with `--check-tensors` and needs a model loaded from a file path; both fail the load with their own message.
- The host computes how many tensors and bytes each worker will read from each file and announces them, which gives the worker's progress lines their totals.
- `GGML_RPC_FILE_LOAD_TIMEOUT` (whole seconds; unset, zero, negative or unparsable means 60): the identity check and each file-load request must be answered within this time plus 1 s per 8 MiB requested, counted from when the request went out. Otherwise the host process aborts (`GGML_ABORT`) with a message containing `rpc_file_load_timeout: endpoint=... command=... timeout_s=...`, prefixed by the source file and line. There is no fallback after a timeout because the connection is out of step. Other RPC calls have no new timeout.
- When any worker advertises direct file load, the host skips its whole-file mmap prefetch, so the workers' shares are not read through the host; it still reads its own share and the ranges hashed for the identity check.
- `llama-server` logs `srv  load_progress: fraction=0.42` at most every 250 ms during a model load, with the final `1.00` exactly once. Its return value and state reporting are unchanged.

**Wire format.** Two new commands, `RPC_CMD_SET_TENSOR_FILE` and `RPC_CMD_FILE_IDENTITY`, and new bits in the existing HELLO response flags:

| Bit | Meaning |
|---|---|
| `1 << 7` | First version of the extension (`4b46c3c8c`, size check only). No longer advertised or recognised. |
| `1 << 6` | Current extension: file identity check plus file load. Set by workers started with `--model-dir`. |
| `1 << 5` | `RPC_DISPATCHER_FLAGS_FILE_LOAD_TOTALS`: the worker accepts announced totals and returns a reason code. Advertised together with `1 << 6`. |

Every command sent on the wire keeps its number and message layout, and the protocol version stays 10.0.0, as upstream.

**Compatibility with stock peers.** A stock worker never sets these bits, so a fork host sends it the weights over the socket, in parallel with the other workers (in required mode the load fails with `no_capability`). A stock host ignores the bits and never sends the new commands, so a fork worker serves it as stock. First-version peers never use file load with current peers; second-version peers (`1 << 6` only, `cb26f3b4d` to `4c2f23594`) interoperate with current ones without totals.

**Files.** `ggml/include/ggml-rpc.h`, `ggml/src/ggml-rpc/ggml-rpc.cpp`, `tools/rpc/rpc-server.cpp`, `src/llama-model-loader.cpp`, `src/llama-model-loader.h`, `src/llama-model.cpp`, `tools/server/server-context.cpp` (load progress line only).

**Does not touch.** Tensor allocation, `--tensor-split`, compute and inference, upstream's existing RPC commands, and the protocol version.

## Branches

| Branch | What it is |
|---|---|
| `potluck` | Default branch and production line: all eight changes, the release workflow and this documentation. Build from this. Called `ps5-direct-pull` until 23 November 2026 (UTC). |
| `master` | An older upstream commit plus four older documentation-only commits. No code changes; not used for builds. |
| `ps5-parallel-upload` | The original single patch for change 1 (`27a46715d`) on the older upstream base `330f0189c` (protocol 8). Kept for reference. |

The former branches `ps5-parallel-upload-head` (`6cff7a968`), `ps5-loader-fixes` (`6cff7a968`, `74ab8e0d0`, `a01665d30`) and `ci/release-builds` (the eight commits above plus `a757db077`, `cc81d31a0`, `8ffb6b6d0`) are not kept: every commit on them is in `potluck`'s history.

## Builds and releases

Release builds of `llama-server` and `rpc-server` are made by this repository's release workflow from `potluck-llama-v*` tags. See [docs/potluck-builds.md](docs/potluck-builds.md) for what a release contains, how to cut one, how to verify a download and how Potluck consumes it.

## Licence

MIT, unchanged from upstream. The [LICENSE](LICENSE) file is kept as upstream ships it.

## Known open questions

- The file identity check samples the model file and does not prove that every tensor arrived intact on every device. Verifying loaded weights (per-tensor digests) is an open research question, tracked in [issue 1](https://github.com/Jtonna/llama.cpp-for-datacenter-gpu-orchestrator/issues/1).
- One real direct load produced wrong model output although the byte counts were identical (`03c9c2820`). Its cause was not established: the sampled identity check cannot tell a wrong read from a fault after loading. The optional load digest was added so that the bytes a worker loaded can be compared across loads and workers.
