# Fork changes

This fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) exists to serve the [ps5-ai-datacenter-manager](https://github.com/Jtonna/ps5-ai-datacenter-manager) project — a self-hostable PS5 LLM cluster where an OMEN desktop orchestrates three jailbroken PS5s running the Vulkan `rpc-server` backend. For general llama.cpp use, go to upstream.

## Why this fork exists

Upstream's RPC client uploads tensor data to each `rpc-server` backend **serially** at model load time. On a 3-PS5 split, that means PS5-A's full shard streams first, then PS5-B's, then PS5-C's — only one TCP connection carrying data at any moment. On measured hardware (OMEN 2.5 GbE → 3 PS5s @ 1 GbE each) this capped aggregate upload at ~95 MB/s per active worker, leaving two PS5 inbound NICs and two-thirds of the OMEN uplink idle during each phase.

This fork adds a **parallel RPC tensor-upload path** in the loader so all three PS5s receive their shards concurrently, saturating the OMEN uplink (~280-295 MB/s aggregate on 2.5 GbE) and cutting cold-load wall time to ~1/2.5 of serial.

## Branches

| Branch | Base | Purpose |
|---|---|---|
| `master` | upstream HEAD (periodic re-sync) | Doc-only — this file + the README banner. No code changes. |
| `ps5-parallel-upload` | `330f018` + one commit | The actual change. **This is what you build from.** |

## Base-commit lock: `330f018`

The patch branch is pinned to upstream commit [`330f018`](https://github.com/ggml-org/llama.cpp/commit/330f018). This is **not** the latest llama.cpp — it's the commit the PS5 fleet's deployed `rpc-server` image was built from.

At `330f018` the RPC wire-protocol major version is `8`. Upstream bumped this to `10` with [PR #16276 (multi-device RPC endpoint)](https://github.com/ggml-org/llama.cpp/pull/16276) and tweaked the loader hash-cache path with [PR #29413 (b8738, "RPC: Improve loading times")](https://github.com/ggml-org/llama.cpp/pull/29413). **A client at major 10 refuses to connect to a server at major 8**, so rebasing this fork onto newer upstream would require simultaneously rebuilding the PS5s' `rpc-server` image + redeploying the PS5 payload — a cross-cutting change we deliberately keep out of scope here.

If you later need to rebase this fork, that's a two-step project: (1) bump base commit on both client AND PS5 server, (2) re-benchmark everything and update the numbers below. Open an issue first.

## The patch (`ps5-parallel-upload` branch)

Single commit on top of `330f018`. Stats: **4 files, +253 / -34**.

| File | Change |
|---|---|
| `src/llama-model-loader.cpp` (+100 / -3) | Persistent per-buffer async upload map (`mmap_async_by_buf`), new `finalize_async_uploads()` method, cancel-flag + mutex-guarded writes for concurrent per-tensor dispatch, `tid=` + per-call duration INFO logs. |
| `src/llama-model-loader.h` (+23) | Member map + mutex + atomic cancel flag + `rpc_set_skip_hash_probe` function pointer. |
| `src/llama-model.cpp` (+36 / -3) | Outer load loop spawns one `std::thread` per `(ctx, bufs)` entry instead of calling `load_all_data` serially; joins all before `finalize_async_uploads()`; rethrows worker exceptions; honors cancel on partial failure. |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` (+66 / -33) | (a) `set_skip_hash_probe` toggle exported via `get_proc_address` so the loader can suppress the per-tensor blocking `SET_TENSOR_HASH` probe on cold loads where no server-side cache entries exist — `cache_flag=0` is forced in the skip path to prevent accidental USB-cache writes on PS5s that re-enable `-c`. (b) `rpc_dispatcher` gains an atomic `faulted` flag so a PS5 crash mid-load surfaces promptly with a clear `RPC dispatcher faulted` message instead of hanging the loader (`GGML_ABORT` from a synchronize that never completes). |

**Not touched:** `ggml-rpc.h`, any `rpc_msg_*` struct, any `RPC_CMD_*` enum, `RPC_PROTO_MAJOR_VERSION`. Wire protocol is byte-identical to `330f018`.

## Semantics

The patch targets the **transport layer** of load, not the tensor allocation. `--tensor-split` still determines which physical PS5 owns which layers, and that same split governs inference distribution — unchanged from stock. The patch only affects *how the bytes get there* during load. After load, inference path is bit-identical; tested via temperature-0 seeded prompts yielding token-for-token identical output on Gemma 12B, Qwen 27B, and Kimi-Dev 72B.

**Scope limits of the MVP patch:**

- **mmap path only.** The `--no-mmap` codepath reuses a shrinking `read_buf` per tensor, which would be unsafe to dispatch fire-and-forget without additional staging. Users running with `--no-mmap` see no speedup. The sync fallback is preserved for them.
- **`check_tensors` falls back to sync.** Same gating logic upstream uses — preserved.
- **Non-RPC backends unaffected.** The new path is gated on `caps.async && caps.events` *without* the `host_buffer` requirement; non-RPC backends with `host_buffer=true` keep the existing pinned-host-staging path.

## Measured impact

All timings on measured hardware: OMEN host (Mediatek MT7922 WiFi 6E, USB-C 2.5 GbE adapter to switch, Ryzen+NVIDIA) + 3 PS5 BC-250 workers (Marvell/NXP IW620 wifi + 1 GbE eth + RDNA 2 GPU @ 2230 MHz pinned), post-tune environment.

| Profile | Stock load | Patched load | Speedup | Decode (both) |
|---|---|---|---|---|
| bench-gemma12b-3-2048 (3-PS5 split, 6.7 GB) | 66.2 s | 28.8 s | **2.30×** | identical, 25.6 tok/s |
| bench-qwen27b-3-2048 (3-PS5 split, 17 GB) | 174.5 s | 73.0 s | **2.39×** | identical, 14.6 tok/s |
| bench-kimi-dev-72b-iq2s-3-2048 (3-PS5 split, 28 GB) | 286.4 s | 114.6 s | **2.50×** | identical, 8.59 tok/s |

Per-PS5 stream rates during a patched 3-way load: ~93-98 MB/s each, aggregate ~280-295 MB/s. The OMEN's 2.5 GbE uplink is now the ceiling, as expected after moving the choke off the per-worker NIC.

## Correctness

The patch must not change inference output. Validated three ways:

1. **Byte-identical response**: temperature 0, fixed seed (42), `cache_prompt: false`, same prompt — sha256 of the whole response object matches stock on every model tested (Gemma 12B, Qwen 27B, Kimi-Dev 72B).
2. **Byte-compatible scheduler log parsing**: the host scheduler (`llm-scheduler`) parses `-lv 4` child output for per-device memory and readiness markers (`common_memory_breakdown_print`, `load_tensors: RPC[N][...] model buffer size = X MiB`, `srv server_run_i: model loaded`, listening line). Order within those groups may shift between runs because of threading, but the grep'd set is identical.
3. **Fault handling**: the new `faulted` flag was added specifically so that a PS5 crashing mid-load (observed in the Devstral one-PS5 OOM case) surfaces as an explicit abort with a tail of the child log, rather than hanging on `ggml_backend_synchronize` because the dispatcher thread exited without setting its promises.

## Build

Standard CMake; same as upstream. Produce the `llama-server` binary (the ps5-ai-datacenter stack installs it as `llama-server-rpc` by convention — that's a rename at install time, not a separate target):

```bash
git checkout ps5-parallel-upload    # pinned at 330f018 + the patch
cmake -B build -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
# Optionally disable embedded UI assets if the 330f018 HF-UI fetcher step fails:
#   -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
```

The scheduler container in ps5-ai-datacenter-manager builds a thin Docker image FROM `llm-scheduler:test` that copies this binary over the stock one, exposed as the tag `llm-scheduler:test-parallel-upload-v3`. See the upstream project's docs for the container swap flow.

## Pending / future work

- **MPTCP prototype** (separate branch, in progress): Bond a wifi path alongside the ethernet path for one PS5 so OMEN can transmit over two physical interfaces concurrently. Would lift the 2.5 GbE OMEN ceiling if measurably effective. Not merged here yet.
- **Async hash probe**: `SET_TENSOR_HASH` probe is currently suppressed wholesale on cold loads via the toggle. A per-tensor async hash probe would benefit warm reloads too, without the current cold-load blocking.
- **`--no-mmap` support**: Requires per-concurrent-backend staging buffers cycled with events. Doable; deferred.
- **Upstream PR**: Worth sending once the fault-handling is polished and the base-commit question (patch against current master vs. against a pinned old commit) is resolved. Our ps5-ai-datacenter fleet needs the pinned-old-commit version, but upstream wants it against current master; these are two different reviews.

## Getting in touch

Issues specific to this fork: file at https://github.com/Jtonna/llama.cpp/issues. For anything else, upstream.
