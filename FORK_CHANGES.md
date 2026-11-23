# Fork changes

This repository is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), made for [Potluck](https://github.com/Jtonna/datacenter-gpu-orchestrator), which splits large models across several machines with `llama-server` and `rpc-server`. For general llama.cpp use, go upstream.

The fork changes three things:

1. **Parallel upload.** The model loader loads every device's share on its own thread, so RPC workers receive their weights at the same time instead of one after another.
2. **Loader fixes.** Two data races that the parallel load introduced (a shared file position without mmap, an unlocked map lookup with mmap) are fixed.
3. **Direct file load.** An `rpc-server` started with `--model-dir` reads its share from its own copy of the model file after a file identity check, with path confinement, bounded waits, a required mode and progress lines.

Upstream base: [`0a76757469ea94ad5f88d14e18cc7370b9931f99`](https://github.com/ggml-org/llama.cpp/commit/0a76757469ea94ad5f88d14e18cc7370b9931f99). The RPC protocol version (10.0.0) is unchanged. Build from the `potluck` branch.

What changed, why, the measurements and the technical reference: [CHANGELOG_FOR_POTLUCK.md](CHANGELOG_FOR_POTLUCK.md).
