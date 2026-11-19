# Fork changes

This fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) exists to serve the [ps5-ai-datacenter-manager](https://github.com/Jtonna/ps5-ai-datacenter-manager) project — a self-hostable PS5 LLM cluster where an OMEN desktop orchestrates three jailbroken PS5s running the Vulkan `rpc-server` backend. For general llama.cpp use, go to upstream.

## Why this fork exists

Upstream's RPC client uploads tensor data to each `rpc-server` backend **serially** at model load time. On a 3-PS5 split, that means PS5-A's full shard streams first, then PS5-B's, then PS5-C's — only one TCP connection carrying data at any moment. On measured hardware (OMEN 2.5 GbE → 3 PS5s @ 1 GbE each) this capped aggregate upload at ~95 MB/s per active worker, leaving two PS5 inbound NICs and two-thirds of the OMEN uplink idle during each phase.

This fork adds a **parallel RPC tensor-upload path** in the loader so all three PS5s receive their shards concurrently, saturating the OMEN uplink (~280-295 MB/s aggregate on 2.5 GbE) and cutting cold-load wall time by ~2.1-2.35×.

## Branches

| Branch | Base | Purpose |
|---|---|---|
| `master` | upstream HEAD (periodic re-sync) | Doc-only — this file + the README banner. No code changes. |
| `ps5-parallel-upload` | `330f018` + one commit | Historical. The original patch against protocol v8, retained for rollback/forensics. |
| `ps5-parallel-upload-head` | upstream HEAD (`0a76757`) + one commit | **Current. Build from this.** |

## Base-commit decision

**Current**: `ps5-parallel-upload-head` is based on upstream HEAD (`0a76757469ea94ad5f88d14e18cc7370b9931f99`, protocol MAJOR 10). The PS5 fleet's deployed `rpc-server` has also been upgraded to this commit — stock Vulkan `rpc-server` works, no fork-of-the-fork needed on the server side.

**Why HEAD, not 330f018 (the old base)**: HEAD alone is 6-13% faster than 330f018 on cold loads (measured across the fleet). More importantly, HEAD includes [`cf92c132` "RPC: Improve loading times"](https://github.com/ggml-org/llama.cpp/commit/cf92c132) which added a server-advertised hash-cache opt-in flag. Our original 330f018-base patch needed a hacky "skip the hash probe" toggle; HEAD solves it natively. HEAD also keeps the original `RPC_STATUS_ASSERT` in `rpc_dispatcher::work()` that the v3 patch temporarily removed — so our v4 "restore fast-abort" hack is unneeded.

**Net**: the HEAD-based patch is **~1/4 the size** of the 330f018-based patch (164 LOC vs 253), and `ggml/src/ggml-rpc/ggml-rpc.cpp` is untouched. Three loader files only.

## The patch (`ps5-parallel-upload-head` branch)

Single commit (SHA varies on re-apply; the one validated against the fleet on 2026-11-19 is `d7dcdaf59` locally, pushed to origin as `6cff7a96`). Stats: **3 files, +164 / -7**.

| File | Change |
|---|---|
| `src/llama-model-loader.cpp` (+111/-4) | Per-buffer async upload map, new `finalize_async_uploads()`, cancel-flag + mutex-guarded writes for concurrent per-tensor dispatch, `tid=` + per-call duration INFO logs. |
| `src/llama-model-loader.h` (+22/-0) | Member map + mutex + atomic cancel flag + `rpc_set_skip_hash_probe` function pointer (vestigial; unused on HEAD where hash cache is server-advertised). |
| `src/llama-model.cpp` (+38/-5) | Outer load loop spawns one `std::thread` per `(ctx, bufs)` instead of calling `load_all_data` serially; joins all before `finalize_async_uploads()`; rethrows worker exceptions; honors cancel on partial failure. |

**Not touched:** `ggml/src/ggml-rpc/` (anywhere), `ggml-rpc.h`, any `rpc_msg_*` struct, any `RPC_CMD_*` enum, `RPC_PROTO_MAJOR_VERSION`. Wire protocol is byte-identical to upstream HEAD.

## Semantics

The patch targets the **transport layer** of load, not the tensor allocation. `--tensor-split` still determines which physical PS5 owns which layers, and that same split governs inference distribution — unchanged from stock. The patch only affects *how the bytes get there* during load.

After load, inference path is bit-identical to stock HEAD. Validated via temperature-0 seeded prompts yielding **byte-identical output on all 16 tested profiles** against the stock-HEAD references, including models that drift between major llama.cpp versions.

**Scope limits of the patch:**

- **mmap path only.** The `--no-mmap` codepath reuses a shrinking `read_buf` per tensor, which would be unsafe to dispatch fire-and-forget without additional staging. Users running with `--no-mmap` see no speedup. The sync fallback is preserved for them.
- **`check_tensors` falls back to sync.** Same gating logic upstream uses — preserved.
- **Non-RPC backends unaffected.** The new path is gated on `caps.async && caps.events` *without* the `host_buffer` requirement; non-RPC backends with `host_buffer=true` keep the existing pinned-host-staging path.

## Measured impact

All timings on measured hardware (OMEN host with Mediatek MT7922 WiFi 6E + USB-C 2.5 GbE adapter to switch + Ryzen+NVIDIA; 3 PS5 BC-250 workers with Marvell/NXP IW620 wifi + 1 GbE eth + RDNA 2 GPU pinned at 2230 MHz).

| Profile | Stock HEAD | Patched HEAD | Speedup |
|---|---|---|---|
| tinyllama (1 PS5) | 6.9 s | 6.1 s | noise |
| qwen4b (1 PS5) | 27.2 s | 27.3 s | 1.00× |
| gemma12b (1 PS5) | 62.3 s | 62.4 s | 1.00× |
| qwen4b (2 PS5) | 26.2 s | 15.0 s | **1.75×** |
| gemma12b (2 PS5) | 62.6 s | 33.6 s | **1.87×** |
| devstral24b (2 PS5) | 130.7 s | 70.9 s | **1.84×** |
| qwen27b (2 PS5) | 159.9 s | 87.7 s | **1.82×** |
| qwen4b (3 PS5) | 28.0 s | 15.5 s | **1.81×** |
| gemma12b (3 PS5) | 62.4-66.4 s | 28.3-32.2 s | **~2.1×** |
| devstral24b (3 PS5) | 129.7 s | 57.2 s | **2.27×** |
| deepseek-v2-lite (3 PS5) | 96.5 s | 41.1 s | **2.35×** |
| qwen30b-a3b (3 PS5) | 178.6 s | 82.7 s | **2.16×** |
| qwen27b (3 PS5) | 157.7 s | 71.9 s | **2.19×** |
| llama33-70b-q2 (3 PS5) | 244.8 s | 114.6 s | **2.14×** |
| kimi-dev-72b-iq2s (3 PS5) | 252.8 s | 115.3 s | **2.19×** |
| qwen25-72b-q2 (3 PS5) | 273.2 s | 128.9 s | **2.12×** |

Per-PS5 stream rates during a patched 3-way load: ~97 MB/s each, aggregate ~290 MB/s. The OMEN's 2.5 GbE uplink is the current ceiling. Decode is bit-identical within noise on all profiles (1-3% delta, within run-to-run variance).

## Correctness

Validated three ways:

1. **Byte-identical response**: temperature 0, fixed seed (42), `cache_prompt: false`, same prompt — sha256 of the whole response object matches stock HEAD on **all 16 profiles** (vs older 330f018 base, Devstral drifted at the last sentence; on HEAD-vs-HEAD comparison the patch is clean).
2. **Byte-compatible scheduler log parsing**: the host scheduler parses `load_tensors: RPC<N>[addr] model buffer size = N MiB` for memory accounting and `srv  server_run_i: model loaded` for readiness. Neither line changes.
3. **Rollback target**: stock `llm-scheduler:test-head` (upstream HEAD + no patch) is preserved as a 1-container-swap rollback.

## Build

Standard CMake; same as upstream. Produce the `llama-server` binary:

```bash
git checkout ps5-parallel-upload-head
cmake -B build -DGGML_RPC=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
      -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build --target llama-server -j
```

The ps5-ai-datacenter-manager stack installs this as `llama-server-rpc` beside the stock `llama-server`. See the upstream project's docs for the container swap flow.

## Hang-on-OOM on the PS5 side

Separately from this patch, we observed that on upstream HEAD the `rpc-server` can hang (rather than crash cleanly) under PS5 memory exhaustion, holding a TCP connection open while the server thread stalls. This is NOT caused by this patch (stock HEAD does the same), but it affects our fleet's operating envelope. We address it with a 4-layer defense system in the ps5-ai-datacenter-manager project (scheduler-side fit checks + hang detection + PS5-side recovery daemon), not here. See that project's benchmarks.md "Hang-on-OOM finding" section.

## Pending / future work

- **Upstream PR**: the HEAD-based patch is a clean candidate for an upstream submission to llama.cpp. Not blocked by anything in this fork; just needs the submit effort.
- **Async hash probe**: on `rpc-server` started with `-c` (cache enabled), the server-side `SET_TENSOR_HASH` probe is still synchronous per-tensor. Low priority because our fleet runs without `-c`; the probe is auto-skipped.
- **`--no-mmap` support**: Requires per-concurrent-backend staging buffers cycled with events. Deferred.
- **MPTCP transport bonding**: separate experiment, not in this patch tree.

## Getting in touch

Issues specific to this fork: file at https://github.com/Jtonna/llama.cpp/issues. For anything else, upstream.
