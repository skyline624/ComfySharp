# Metadata inspection

```sh
dotnet run --project tools/ComfySharp.ModelInspect -- /absolute/path/weights.safetensors
dotnet run --project tools/ComfySharp.ModelInspect -- /absolute/path/weights.safetensors --tensors 20
dotnet run --project tools/ComfySharp.ModelInspect -- /absolute/path/weights.safetensors --sha256
```

The CLI requires one explicit file and never scans directories or downloads files. The default output contains aggregate size, dtype and rank counts; it excludes file paths, tensor names and metadata values. `--tensors <1..1000>` includes that many names and shapes. `--sha256` explicitly opts into reading the entire file; otherwise only the header is read.

The parser validates the header, shape arithmetic, byte ranges and complete indexing. A successful metadata inspection is separate from weight or model-family compatibility. `readerUnsupportedDTypes` identifies valid header dtypes that the current TorchSharp API cannot materialize; absence of those dtypes does not prove a model is executable. Unknown formats produce an explicit error.

The project references Inference without a libtorch CPU/CUDA package. Metadata inspection does not initialize native Torch code. Header size, tensor count and rank have the reader's default limits; metadata inspection permits any file-bounded tensor size because it does not allocate tensor storage. Tensor materialization has separate memory limits.

Results are JSON on stdout. Errors are JSON on stderr with exit code 1; argument errors use 2 and cancellation uses 130. Optional SHA-256 honors Ctrl+C. Header parsing is bounded to 16 MiB.
