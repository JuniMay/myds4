# myds4

Local inference on Apple silicon, with a C/Metal engine and a Rust HTTP
server. The supported models are DeepSeek V4.1 Flash, Qwen3.8-27B NVFP4,
Qwen3.8 Flash-Next with NVFP4 experts and MXFP8 projections, and
Nemotron 3.5 Transcribe in NVFP4 with its optional BF16 DFlash2 drafter.

The server can serve these architectures in one process. It registers their
metadata at startup and loads each model's weights on its first inference
request. Every model has its own request queue and state. All models share
the memory that other programs leave, less a reserve kept available to
them, and one KV-cache budget; an optional model-memory limit covers all
models and excludes KV (see [Usage](docs/usage.md#memory)).

## Build

The engine uses Metal 4.1 tensor operations and requires a compatible Apple
silicon Mac, macOS 27, Xcode command-line tools, a C23 compiler and Rust.
Install the native dependencies, then build:

```sh
brew install pkgconf yyjson jpeg-turbo
make
make server
```

`build/myds4` is the CLI; `build/myds4-server` is the server. Python is used
for optional reference checks and benchmarks, not for inference. Weights
are read from the configured checkpoints. DeepSeek's Engram lookup is generated
from its tokenizer when the model loads.

## Serve models

Put the server settings in `~/.config/myds4/server.json`. Replace the paths
below with your checkpoint and existing API-key paths:

```json
{
  "port": 8008,
  "host": "127.0.0.1",
  "api_key_file": "/path/to/api-key",
  "tui": true,
  "memory": 80,
  "reserve": 10,
  "kv_cache": 24,
  "models": [
    {
      "model": "/path/to/DeepSeek-V4.1-Flash-1017",
      "ctx": 1048576,
      "batch": 4,
      "temperature": 1,
      "top_p": 1
    },
    {
      "model": "/path/to/Qwen3.8-27B-NVFP4",
      "ctx": 262144,
      "batch": 4,
      "draft": 1,
      "temperature": 1,
      "top_p": 1,
      "top_k": 0
    }
  ]
}
```

An API key must contain 16–256 characters. To create a new key, use
`openssl rand -hex 32` and save it in a file readable only by your account.
Start the server with:

```sh
build/myds4-server
```

The API base URL is `http://127.0.0.1:8008/v1`. Fetch `/v1/models` for the
model names and send the desired name in each Chat Completions request.
`--config FILE` selects another configuration; server flags override file
settings. Changes take effect after a restart.

For one CLI request:

```sh
build/myds4 -m /path/to/checkpoint -p "Explain the code." -t 1 --top-p 1
```

## Documentation

- [Usage](docs/usage.md): configuration, API, CLI and terminal controls.
- [Design](docs/design.md): model execution, ownership and numerical behavior.
- [Testing](docs/testing.md): CPU, GPU and model checks.
- [Performance](docs/performance.md): benchmark methods and interpretation.
