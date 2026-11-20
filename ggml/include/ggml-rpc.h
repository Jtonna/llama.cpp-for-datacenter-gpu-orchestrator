#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    10
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    0

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 99, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);

GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(
        const char * endpoint,
        const char * cache_dir,
        size_t n_threads,
        size_t n_devices,
        ggml_backend_dev_t * devices);

// fork extension: model_dir enables RPC_CMD_SET_TENSOR_FILE, restricted to files inside that directory (NULL = disabled)
GGML_BACKEND_API void ggml_backend_rpc_start_server_ext(
        const char * endpoint,
        const char * cache_dir,
        const char * model_dir,
        size_t n_threads,
        size_t n_devices,
        ggml_backend_dev_t * devices);

// fork extension: let the RPC server owning tensor->buffer read the bytes from its own copy of `path`, once that copy
// was found identical to `path` (size + header/metadata + sampled blocks, checked on the first call per server and path).
// Returns false without writing anything usable if unsupported/refused/different; the caller must then upload the data itself.
// Aborts if the server does not answer within GGML_RPC_FILE_LOAD_TIMEOUT (default 60) s + 1 s per 8 MiB.
GGML_BACKEND_API bool ggml_backend_rpc_buffer_set_tensor_from_file(struct ggml_tensor * tensor, const char * path,
        uint64_t file_offset, uint64_t file_size, size_t offset, size_t size);

// fork extension: true if buft is an RPC buffer type whose server accepts set_tensor_from_file
GGML_BACKEND_API bool ggml_backend_rpc_buft_supports_set_tensor_file(ggml_backend_buffer_type_t buft);

// fork extension: endpoint ("host:port") of an RPC buffer type, NULL if buft is not an RPC buffer type
GGML_BACKEND_API const char * ggml_backend_rpc_buft_endpoint(ggml_backend_buffer_type_t buft);

// fork extension: result of ggml_backend_rpc_buft_prepare_file_load
enum ggml_rpc_file_load_status {
    GGML_RPC_FILE_LOAD_OK                = 0, // the server's copy matches; set_tensor_from_file may be used
    GGML_RPC_FILE_LOAD_NO_CAPABILITY     = 1, // not an RPC buffer type, or the server does not advertise the extension
    GGML_RPC_FILE_LOAD_REFUSED           = 2, // the server refused (no such file in its --model-dir, unreadable, ...)
    GGML_RPC_FILE_LOAD_IDENTITY_MISMATCH = 3, // the server's copy differs (size or sampled content)
};

// fork extension: verify (FILE_IDENTITY) the server's copy of `path` before any set_tensor_from_file for it, and
// announce how many tensors / bytes the client will ask it to read from that file, so the server can log totals.
// The announcement is only sent to servers that advertise it; others get the plain identity check. The verdict is
// cached per server and path (later calls and set_tensor_from_file reuse it). Call at most once per server and path.
GGML_BACKEND_API enum ggml_rpc_file_load_status ggml_backend_rpc_buft_prepare_file_load(ggml_backend_buffer_type_t buft,
        const char * path, uint64_t file_size, uint64_t n_tensors, uint64_t n_bytes);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
