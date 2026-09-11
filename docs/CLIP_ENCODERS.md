# CLIP text encoders and conditioning

The inference library now contains a parameterized CLIP text transformer, a strict safetensors weight adapter and ComfyUI conditioning wrappers. These are experimental CPU/F32 components. No pretrained encoder, SD1/SDXL generation workflow or GPU backend is qualified by their presence.

The functional source is [ComfyUI at `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a): `comfy/clip_model.py`, `comfy/sd1_clip.py`, `comfy/sdxl_clip.py`, the basic attention function in `comfy/ldm/modules/attention.py`, and key conversions in `comfy/utils.py`. Python is confined to a separate reference laboratory; the application, diagnostic and distributed tests execute C# and native libraries.

## Computation and ownership

| Configuration | Hidden / MLP width | Layers / heads | Activation | Learned tensors / parameters |
|---|---:|---:|---|---:|
| CLIP-L | 768 / 3072 | 12 / 12 | QuickGELU | 197 / 123,650,304 |
| CLIP-G | 1280 / 5120 | 32 / 20 | exact GELU | 517 / 694,659,840 |

Both use the 49,408-entry vocabulary and exactly 77 learned positions. The counts include the square pooled projection. The graph performs embedding lookup, position addition, pre-norm residual attention and MLP blocks, final normalization and pooling. Attention follows the source `attention_basic` order: flattened batch/head dot products, scaling, additive mask, softmax and value product. It is not an SDPA substitution.

`ClipWeightSet` owns a shared immutable bank; `ClipTextEncoder` and conditioning wrappers retain independent owners. Each forward retains its bank until completion. Outputs own native tensors and survive model or ambient-scope disposal. Callers must dispose each result. Temporary tensors are scoped per layer. Cancellation is checked between loading steps and layers; an in-progress native operation finishes before cancellation can be observed. The graph is frozen inference with no autograd contract.

Weights may be stored as F32, F16 or BF16 and are converted to CPU F32. Resident learned parameters alone require 494,601,216 bytes for L or 2,778,639,360 bytes for G. The metadata plan estimates additional per-source loading space; neither estimate includes activations, allocator caches, runtime libraries or process memory. A bank is populated directly without allocating a second initialized model.

## Source contracts

- SD1-L selects normalized final hidden states and **unprojected** pooled output.
- SDXL-L and SDXL-G select unnormalized penultimate hidden states by default. Composed SDXL concatenates L before G, cuts to the shorter sequence and returns G's projected pooled output. L's projection is unused by composition.
- Padding masks count the source's valid tokens, including its left-padding behavior. Pooling uses `count - 1`, with zero selecting the last position. Without counts, the low-level transformer selects the first EOS, or position zero when EOS is absent.
- Prompt weights apply after the transformer against an encoded empty sequence. Sections concatenate on the sequence axis; pooled output comes from the first section. Optional attention masks flatten the original sections.
- Scalar layer selection and all-layer outputs preserve the source's supported operations. In the unusual all-layer case, weighting indexes layers using the first token weights; it is not ordinary per-token weighting. Rank-four zero-mask multiplication must satisfy the source's in-place broadcasting constraints. SDXL all-layer composition uses the source's literal axis-one cut.
- Unsupported row lengths, token IDs, shapes, selectors and incompatible broadcasts receive explicit diagnostics. Composed SDXL rejects branch `ReturnAttentionMasks`: the source wrapper cannot unpack that extra result. Unknown document nodes and future embedding resolvers are not replaced with fabricated conditioning.

## Explicit checkpoint layouts

`ClipCheckpointLoader.Inspect` validates descriptors before native initialization and binds its immutable plan to the same open `SafeTensorFile` later loaded. It validates every selected name, shape and dtype. Missing parameters, alias collisions, unexpected encoder parameters and quantized layouts fail explicitly. Components outside a named encoder prefix are ignored. Canonical and standalone OpenCLIP layouts select the entire tensor namespace and therefore reject unrelated components.

| CLI layout | Selected source namespace |
|---|---|
| `canonical` | `text_model.*`, `text_projection.weight` |
| `clip-l` / `clip-g` | `clip_l.transformer.*` / `clip_g.transformer.*` |
| `sd1` | `cond_stage_model.transformer.*`, including the source's older embeddings/encoder names |
| `sdxl-l` | `conditioner.embedders.0.transformer.*` |
| `sdxl-g` | `conditioner.embedders.1.model.*`, with OpenCLIP conversions |
| `openclip` | standalone OpenCLIP text names |

OpenCLIP combined Q/K/V weights and biases split in Q, K, V order along axis zero. Unsuffixed `text_projection` transposes; `text_projection.weight` is already in canonical `[out,in]` order. Known unused `logit_scale` and position IDs are recorded as ignored. A missing L projection is acceptable only for an explicitly unprojected path. G projection remains required for its announced profile. These adapters do not claim safe loading of pickle, quantization, adapters or arbitrary architectures.

## Local diagnostic

Run from the repository root with a checkpoint selected explicitly by the operator:

```text
dotnet run -c Release --project tools/ComfySharp.RuntimeProbe -- clip --weights checkpoint.safetensors --layout sd1 --profile sd1-l --inspect
dotnet run -c Release --project tools/ComfySharp.RuntimeProbe -- clip --weights checkpoint.safetensors --layout sd1 --profile sd1-l --text "a cat" --repeat 3 --sha256
```

Inspection requires no prompt and initializes no native runtime. Hashing is opt-in and reads the same open file. Encoding returns output dimensions, dtypes, hashes, small numeric samples and process observations. The JSON omits checkpoint paths, prompt text, token IDs and arbitrary metadata. `status: ok` reports that the operation ran; `modelCompatibility: not_assessed` remains explicit. `--unprojected-pooled` selects an unprojected L path, including an L checkpoint lacking its otherwise unused projection. Ctrl+C cancels and releases owned resources; jobs are not replayed.

The committed small fixtures can exercise the whole diagnostic without pretrained weights:

```text
dotnet run -c Release --project tools/ComfySharp.RuntimeProbe -- clip --weights tests/ComfySharp.Inference.Tests/Fixtures/clip-tiny-l.synthetic.safetensors --layout canonical --profile sd1-l --synthetic-config 8,20,3,2,quick_gelu --text "a (cat:1.5)" --repeat 3
```

`--synthetic-config` changes the explicitly selected dimensions; it does not create weights, download anything or make a small fixture a pretrained model. Stock configurations are the default.

## Numerical acceptance and current evidence

The profile `clip-basic-cpu-f32-v1` was fixed before acceptance: finite error must satisfy `abs(actual - expected) <= 3e-5 + 3e-5 * abs(expected)`. Nonfinite classification and infinity sign must match; NaN payload/sign are not compared. The reference laboratory uses Python 3.12.10 and PyTorch 2.13.0+cu130, forced CPU/F32, one thread, no-grad and no TF32. Product execution uses TorchSharp 0.107.0 / libtorch 2.10. This is explicitly a comparison across runtime versions.

The laboratory executes the exact frozen class and function ASTs. Its documented CPU operation adapter uses PyTorch Linear/LayerNorm/Embedding without random initialization, forwards embedding `out_dtype` and `cast_to`, selects the actual source `attention_basic`, and forces the intermediate device to CPU. It assigns deterministic, nonuniform parameters before any forward. Expected outputs never come from the C# implementation.

| Reference artifact | SHA-256 |
|---|---|
| 46 reduced-config cases, `clip-encoders.cpu-f32.json` | `80a36961770ef3b86065f95f73899d75bb6a4d3c95dbb266e4941f4d286e1250` |
| Synthetic L input bank, H8/M20/N3/heads2 | `702aff7540e0b417a9f81a36c3978d0e424006baf44d8fcc29f6e79bd8490e1c` |
| Synthetic G input bank, H12/M28/N4/heads3 | `547c956eb83c446d42efbc43968360da2284aea30fe878f46eb86e0282a42917` |
| Two full stock-dimension references, `clip-stock.cpu-f32.json` | `edc3470a883c96f79e75d4c222b2ba8089ba4d4f78de0d5b941dc03175135f6c` |

The reduced corpus and its provenance test pass locally (47 tests). Full stock L/G dimensions each pass three complete forwards against all expected elements, with a separate provenance test (3 tests). Cross-platform encoder qualification remains pending. The full-size tests construct the same deterministic parameters locally, without shipping gigabytes of test weights.

The [three-platform campaigns and trace investigation](qualification/clip-cpu-investigation-5810a4f.md) pass on Windows. Linux and macOS pass the routine and reduced reference tests, but both stock encoders exceed the same fixed numerical tolerance. Their parameter hashes and embeddings match; the first observed difference is in the reconstructed diagnostic LayerNorm. A separate source-versus-source dispatch experiment also exceeds the full L bound without C#. That finding does not qualify the other platforms or establish a new profile. The Windows reference is retained, no failing assertion is skipped and no tolerance is relaxed.

The initial stock run failed the fixed tolerance. An independent trace comparison established that all 714 learned tensors, embeddings and first layer normalization matched exactly. The first rank-three linear projections differed: TorchSharp's C++ functional wrapper used matrix multiplication followed by bias addition, while Python's `aten::linear` used fused `addmm` for contiguous three-dimensional input. Flattening that input to two dimensions before the native linear call restores the source's fused path, then reshaping restores the original axes. The isolated Q/K/V projections became bit-identical, and the full stock tests then passed. The corpus, parameter recipe and tolerance were preserved throughout; the small fixture suite alone had not detected this accumulated rounding difference.

This dispatch distinction is visible in the pinned [LibTorch 2.10 C++ frontend](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/torch/csrc/api/include/torch/nn/functional/linear.h) and [ATen 2.10](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/Linear.cpp); the [2.13 reference ATen implementation](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/Linear.cpp) uses the same relevant fusion condition. It is a frontend dispatch difference rather than a newly introduced 2.13 layer. The [inspected TorchSharp wrapper](https://github.com/dotnet/TorchSharp/blob/8f4def03b641b6753f18076aa5438f8eaaef2d30/src/Native/LibTorchSharp/THSNN.cpp) explains the call chain. That immutable repository revision is not claimed as the build commit of NuGet 0.107.0, whose package metadata omits that commit; direct probes separately establish the installed binary's behavior.

Pretrained checkpoint comparisons, CLIPTextEncode integration with denoisers, textual inversion, patches/LoRA, GPU/offload, training and real SD1/SDXL workflows remain required work. Process memory observations, especially unsupported macOS private-byte measurements, are not proof of no leak. The complete V1 scope and platform requirements remain unchanged.
