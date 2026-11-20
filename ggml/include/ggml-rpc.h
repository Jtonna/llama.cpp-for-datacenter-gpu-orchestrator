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

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
