# Third-party provenance

`LoraFileLoader.cs` adapts the LoRA selection responsibilities of frozen
ComfyUI `comfy/lora.py` and `LoRAAdapter.load` in `comfy/weight_adapter/lora.py`,
under the GPLv3 provenance below. It uses the existing strict safetensors reader,
owned native snapshots and explicit unsupported-key diagnostics. See
[LORA_FILES.md](docs/LORA_FILES.md) for its scope and reference laboratory.

`LoraMath.cs` adapts `LoRAAdapter.calculate_weight` and `weight_decompose`
from the frozen ComfyUI `comfy/weight_adapter/lora.py` and `base.py`, under
the ComfyUI GPLv3 provenance below. `LoraWeightPatch` and immutable-bank
integration provide C# ownership semantics. The separate source collector
also executes `LoraDiff` for gradient comparison; see
[LORA_FOUNDATIONS.md](docs/LORA_FOUNDATIONS.md).

`SdInpaintMask.cs`, `SdInpaintImage.cs` and the two inpaint node routes adapt
the frozen ComfyUI `nodes.py`, `comfy/utils.py`, `comfy/samplers.py` and
`comfy/model_sampling.py`, under the GPLv3 provenance below. Their source
collector and limits are recorded in [SD_INPAINT.md](docs/SD_INPAINT.md).
`inpaint-input.png` is an original procedural RGBA test pattern under the
project GPLv3 licence; it contains no external artwork or model data.

`SdDpmpp2MSampler.cs` adapts the frozen ComfyUI k-diffusion
`sample_dpmpp_2m` implementation. The retained k-diffusion MIT notice below
applies alongside ComfyUI's GPLv3 provenance. Its source-function laboratory
and implementation scope are recorded in [DPMPLUSPLUS_2M.md](docs/DPMPLUSPLUS_2M.md).

`SdScheduler.cs` adapts the nine scheduler routes and `KSampler.set_steps`
from the frozen ComfyUI `comfy/samplers.py`, under the ComfyUI GPLv3 provenance
below. Its Karras/exponential generators retain the k-diffusion MIT notice.
The fixed Beta(0.6,0.6) quantile calculation is an original C# binomial-series
and bisection implementation, not a copy of SciPy. SciPy is used only to run
the source reference in the separate [scheduler laboratory](labs/sd-schedulers-source/README.md).

`SdHeunSampler.cs` ports `sample_heun` with its default `s_churn=0` and the
`to_d` / `append_dims` expression from the frozen ComfyUI k-diffusion sources.
The k-diffusion MIT notice retained below also applies to this adaptation.
The independent source-function collector is described in [HEUN.md](docs/HEUN.md).

`ImageInputNodes.cs` adapts `LoadImage` from the frozen ComfyUI `nodes.py` and
its image/mask contract from `comfy_api/latest/_input_impl/video_types.py`, under
the ComfyUI GPLv3 provenance below. `NativeImageDecoder.cs` uses SkiaSharp and
its native assets (including Linux.NoDependencies), version 3.119.4, under
their existing BSD/MIT notices. `Png16Decoder.cs` is an original C# PNG sample
decoder. The nine `tests/ComfySharp.Host.Tests/Fixtures/input-*` images are
original small codec test patterns created for ComfySharp and distributed
under this project's GPLv3 licence. See [LOAD_IMAGE.md](docs/LOAD_IMAGE.md)
for the implementation scope and remaining source parity work.

ComfySharp is an independent C# port of ComfyUI. It is not an official Comfy-Org release.

Functional reference: [ComfyUI at 1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a), copyright ComfyUI contributors, GNU GPL version 3. The complete upstream GPL text is preserved in LICENSE.

Editor reference: [ComfyUI_frontend v1.51.10 at e7d1c7fc6823e330fdab524610b0000394cb1dbc](https://github.com/Comfy-Org/ComfyUI_frontend/tree/e7d1c7fc6823e330fdab524610b0000394cb1dbc). C# implementations cite their corresponding reference files. No compiled upstream frontend is distributed.

PNG workflow metadata extraction follows that frontend's `src/scripts/metadata/png.ts` and `src/scripts/app.ts`. The exact 223-byte `src/scripts/metadata/__fixtures__/with_metadata.png` is retained as base64 in `tests/Shared/PngMetadataFixture.cs`, copyright ComfyUI_frontend contributors, GPL-3.0-only. Its SHA-256 and supported import behavior are recorded in [PNG_WORKFLOW_IMPORT.md](docs/PNG_WORKFLOW_IMPORT.md).

`ApiPromptImport.cs` adapts the API-to-graph responsibilities of `isApiJson` / `loadApiJson` in that same frozen frontend `src/scripts/app.ts`, copyright ComfyUI_frontend contributors, GPL-3.0-only. Link classification follows the frozen backend's `comfy_execution/graph_utils.py` under its GPLv3 provenance above. [API_PROMPT_IMPORT.md](docs/API_PROMPT_IMPORT.md) records the C# changes: exact string IDs, named literal bindings, retained extension fields, stable initial layout and explicit diagnostics instead of extension callbacks. No JavaScript or Python implementation is distributed by this addition.

`ImportJson.cs` adapts that frontend's `src/utils/jsonUtil.ts` nonfinite-token fallback under the same GPL-3.0-only provenance, replacing its regular expression with a linear C# scanner. The warning text is retained, and Desktop displays it. `PngWorkflowImport.cs` also adapts the workflow-to-prompt fallback order from `handleFile` in `src/scripts/app.ts`. [IMPORT_JSON.md](docs/IMPORT_JSON.md) records the import boundaries and limits.

The static mute/bypass resolver adapts `ExecutableNodeDTO.ts` and `LiteGraphGlobal.isValidConnection` from the pinned frontend's `src/lib/litegraph` tree, alongside `graphToPrompt` from `src/utils/executionUtil.ts`. Node duplication adapts the selection/link reconstruction in `LGraphCanvas.ts` and reference clearing in `LGraphNode.clone`; [WORKFLOW_DUPLICATION.md](docs/WORKFLOW_DUPLICATION.md) describes the C# behavior, unknown-node retention and remaining clipboard/subgraph/reroute work. The complete [LiteGraph MIT notice](docs/licenses/LiteGraph-MIT.txt), **Copyright (C) 2013 by Javi Agenjo**, is preserved byte-for-byte from that commit's [`src/lib/litegraph/LICENSE`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/e7d1c7fc6823e330fdab524610b0000394cb1dbc/src/lib/litegraph/LICENSE), SHA-256 `1b38cc30fe774bcc77538ab99eb460f4e1c8a47c4fddc6e991ee3b5cc2a96bb9`. The frontend contributor/GPL provenance above also remains applicable. [WORKFLOW_MODES.md](docs/WORKFLOW_MODES.md) records the C# changes and the unported virtual/subgraph semantics.

The sigma generators in `src/ComfySharp.Inference/SigmaSchedules.cs` and the no-churn Euler trajectory in `src/ComfySharp.Inference/SdEulerSampler.cs` are ported from the frozen ComfyUI `comfy/k_diffusion/sampling.py` (with `append_dims` from `comfy/k_diffusion/utils.py`). Their numeric references are recorded in `docs/SIGMA_SCHEDULES.md` and `docs/qualification/sd-euler-native210-cpu-f32-v1.md`. The original k-diffusion project's notice, copyright (c) 2022 Katherine Crowson, is retained in [k-diffusion-MIT.txt](docs/licenses/k-diffusion-MIT.txt), copied from [this pinned licence source](https://github.com/crowsonkb/k-diffusion/blob/4601bf085320592473f681a62808ed873d17fad5/LICENSE). This notice is preserved alongside the ComfyUI provenance and project GPLv3 licence; it does not change the frozen functional reference.

The [cross-document clipboard](docs/WORKFLOW_CLIPBOARD.md) reuses the same pinned LiteGraph selection/link semantics and MIT notice. It introduces a ComfySharp text envelope and Avalonia native clipboard adapter, without executing JavaScript or copying browser localStorage. Unknown-node data retention and cross-version conversion limits are documented separately from upstream clipboard behavior.

Selection deletion and cut adapt `LGraphCanvas.deleteSelected`, `LGraphNode.connectInputToOutput` and node removal in `LGraph.ts` from the same frozen frontend, under the LiteGraph MIT and frontend provenance above. [WORKFLOW_DELETE_CUT.md](docs/WORKFLOW_DELETE_CUT.md) records reconnection, protected-node handling, clipboard preflight and unported lifecycles.

Group geometry and editing adapt the frozen frontend's `LGraphGroup.ts`, standard node bounds from `LGraphNode.measure`, and group counter/serialization placement in `LGraph.ts`, under the same LiteGraph MIT and frontend provenance. [WORKFLOW_GROUPS.md](docs/WORKFLOW_GROUPS.md) records the native measured-bounds integration, data preservation and remaining layout/lifecycle differences.

## PreviewAny and tensor text

`src/ComfySharp.Inference/TensorPreviewFormatter.cs` adapts the dense real-number formatting logic in PyTorch's [`torch/_tensor_str.py` at cf30153c4c131c8164ee7798e5022d810682e2cb](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/_tensor_str.py): numeric width and notation, scalar/vector/multidimensional layout, summarized edges and suffix placement. The complete PyTorch copyright statements, redistribution conditions and disclaimer are retained in [PyTorch-BSD-style.txt](docs/licenses/PyTorch-BSD-style.txt), copied from [LICENSE at the same commit](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/LICENSE). The C# adaptation has an explicit supported representation scope; it does not include the Python module or claim to port every PyTorch tensor representation.

The isolated fixture laboratory reports `torch.version.git_version = cf30153c4c131c8164ee7798e5022d810682e2cb` and `torch.__version__ = 2.13.0+cu130`. Its installed `_tensor_str.py` was compared with the official file at that commit and is identical after CRLF-to-LF normalization (normalized UTF-8 SHA-256 `5fcb2c4e80feb8f7d458074a808c00cf0fdc2a67b964a11e7b5ece7bdd135d87`). The CPU output references are preserved in [preview-any.cpu.json](tests/ComfySharp.Inference.Tests/Fixtures/preview-any.cpu.json). This provenance identifies the formatting reference and laboratory only; ComfySharp's libtorch runtime remains the separately locked 2.10 dependency, and the distributed tests do not execute Python.

The `PreviewAny` node in `src/ComfySharp.Nodes.Tensor/TensorNodes.cs` follows [ComfyUI's frozen `comfy_extras/nodes_preview_any.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_preview_any.py), under the project GPLv3 provenance above. That file explicitly credits the original Display Any implementation in **rgthree/rgthree-comfy**. We preserve that attribution to **Regis Gaughan, III (rgthree), copyright (c) 2023**, and its complete MIT notice in [rgthree-comfy-MIT.txt](docs/licenses/rgthree-comfy-MIT.txt). The identified upstream [Display Any source](https://github.com/rgthree/rgthree-comfy/blob/5288408220180af41ce50b0d29135e1ef5f83fdb/py/display_any.py) and [MIT licence](https://github.com/rgthree/rgthree-comfy/blob/5288408220180af41ce50b0d29135e1ef5f83fdb/LICENSE) are pinned at `5288408220180af41ce50b0d29135e1ef5f83fdb` to make this credit verifiable; ComfyUI's frozen node remains the functional reference. No third-party Python or JavaScript extension is executed by this port.

## SD U-Net, classical VAE and sampling

The SD1/SD2 U-Net, classical VAE, discrete sampling and classifier-free guidance ports follow the same frozen ComfyUI GPLv3 reference identified above:

- U-Net and attention: [`comfy/ldm/modules/diffusionmodules/openaimodel.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/ldm/modules/diffusionmodules/openaimodel.py), [`comfy/ldm/modules/attention.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/ldm/modules/attention.py) and [`diffusionmodules/util.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/ldm/modules/diffusionmodules/util.py).
- VAE encoder, decoder and posterior: [`diffusionmodules/model.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/ldm/modules/diffusionmodules/model.py), [`comfy/ldm/models/autoencoder.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/ldm/models/autoencoder.py) and [`distributions/distributions.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/ldm/modules/distributions/distributions.py); image conversion follows [`comfy/sd.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd.py).
- Prediction conversion, conditioning and latent scaling: [`comfy/model_sampling.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/model_sampling.py), [`comfy/samplers.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/samplers.py), [`comfy/conds.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/conds.py) and [`comfy/latent_formats.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/latent_formats.py).

These sources retain earlier diffusion and autoencoder code. The frozen `util.py` explicitly credits OpenAI improved-diffusion, OpenAI guided-diffusion and Phil Wang's denoising-diffusion-pytorch; `model.py` credits pytorch_diffusion and its derived encoder/decoder. The CompVis latent-diffusion and taming-transformers implementations, and Stability AI's generative-models autoencoder, are also identified antecedents. Their respective MIT copyright notices and terms are preserved below. The linked revisions make these source notices verifiable; they do not assert the exact revision originally imported by ComfyUI or replace the frozen functional reference.

| Source notice | Copyright notice |
| --- | --- |
| [CompVis latent-diffusion, LICENSE at a506df5756472e2ebaf9078affdde2c4f1502cd4](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/LICENSE) | Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich |
| [CompVis taming-transformers, License.txt at 3ba01b241669f5ade541ce990f7650a3b8f65318](https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/License.txt) | Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer |
| [pesser/pytorch_diffusion, LICENSE.md at 304bdff2196db604fb66108b0bb2d4a19058b20f](https://github.com/pesser/pytorch_diffusion/blob/304bdff2196db604fb66108b0bb2d4a19058b20f/LICENSE.md) | Copyright (c) 2023 Patrick Esser |
| [Stability AI generative-models, LICENSE-CODE at e8cd657656fa5d61688191730d0e03242bf4ed44](https://github.com/Stability-AI/generative-models/blob/e8cd657656fa5d61688191730d0e03242bf4ed44/LICENSE-CODE) | Copyright (c) 2023 Stability AI |
| [OpenAI guided-diffusion, LICENSE at 0ba878e517b276c45d1195eb29f6f5f72659a05b](https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/LICENSE) and [improved-diffusion, LICENSE at 1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5](https://github.com/openai/improved-diffusion/blob/1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5/LICENSE) | Copyright (c) 2021 OpenAI |
| [Phil Wang's denoising-diffusion-pytorch, LICENSE at 7706bdfc6f527f58d33f84b7b522e61e6e3164b3](https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/LICENSE) | Copyright (c) 2020 Phil Wang |

The following MIT permission and disclaimer apply to each respective copyright notice above:

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The optional [SD source laboratory](labs/sd-source/README.md) executes selected, hash-verified declarations from that frozen ComfyUI source snapshot, retaining this same provenance. It uses the isolated Python/PyTorch/NumPy/einops dependency closure of the CLIP source laboratory. Those laboratory dependencies and the external source snapshot are not included in the application or distributed .NET tests. The checked-in synthetic inputs and independent reference outputs contain no pretrained model weights. These code notices do not assign a licence to any separately obtained model checkpoint.

## Distribution dependencies

The C# CLIP transformer, checkpoint key adapters and conditioning wrappers follow the frozen ComfyUI `comfy/clip_model.py`, `comfy/ldm/modules/attention.py`, `comfy/sd1_clip.py`, `comfy/sdxl_clip.py` and `comfy/utils.py` under the ComfyUI GPLv3 provenance above. [CLIP_ENCODERS.md](docs/CLIP_ENCODERS.md) records source, operation and numerical profiles. The small safetensors fixtures are deterministically generated synthetic inputs, not third-party pretrained weights; their independent reference outputs do not imply model-family compatibility. No Python implementation or interpreter is shipped.

Dependencies retain their own licences: .NET/ASP.NET Core, Avalonia, Nodify.Avalonia and TorchSharp (MIT); libtorch (BSD-style, with additional bundled notices); SQLite (public domain), Microsoft.Data.Sqlite (MIT), SQLitePCLRaw (Apache-2.0); Skia/SkiaSharp (BSD/MIT notices). Exact transitive packages are recorded in packages.lock.json. Packaging must copy every applicable native and managed dependency notice and produce an SBOM before release.

FFmpeg and ANGLE are planned dependencies, not bundled by this bootstrap. Their build configuration, licence obligations and notices must be recorded when integrated. Model weights are not included. The CLIP text resources now included are identified below; each further compatibility record must identify its licence, origin and SHA-256.

Binary packaging is currently local validation only: dependency notice collection
and the exact distribution SBOM are incomplete, so the portable workflow does
not upload binaries. This summary is not a complete distribution notice bundle.
See [packaging requirements](docs/packaging.md) for the unresolved release gate.

## CLIP text tokenization and Unicode data

The four embedded files in `src/ComfySharp.Tokenization/Resources/Clip` are the exact canonical Git blobs from [ComfyUI's frozen `comfy/sd1_tokenizer`](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd1_tokenizer). Their hashes and configuration are recorded in [CLIP_TOKENIZATION.md](docs/CLIP_TOKENIZATION.md). The prompt weighting and SD1/SDXL packing adaptation follows `comfy/sd1_clip.py` and `comfy/sdxl_clip.py` under the ComfyUI GPLv3 provenance above.

All canonical CLIP vocabulary IDs and ordered merges were verified against the resource and vocabulary construction in [OpenAI CLIP at `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6). **Copyright (c) 2021 OpenAI**. Its full [MIT notice](docs/licenses/OpenAI-CLIP-MIT.txt) is preserved from [that commit's LICENSE](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/LICENSE). The configuration identifies the official Hugging Face `openai/clip-vit-large-patch14` repository; this name is attribution, not a remote runtime dependency. No model weights are included, and this notice does not assign licenses to them.

The selected normalization, BPE, special-token and decode behavior references [Transformers CLIP at `a08ace4bbd97e721c98751deec37d87b026acadc`](https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/models/clip/tokenization_clip.py). The source credit is **Copyright 2021 The Open AI Team Authors and The HuggingFace Inc. team**. This is a C# adaptation; the original Python is not distributed. The complete [Transformers Apache-2.0 license](docs/licenses/Transformers-Apache-2.0.txt) is preserved from [the same source commit](https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/LICENSE). Backend behavior also references [Hugging Face tokenizers 0.22.2 at `f383101a26663708484cac0727792aad74f78234`](https://github.com/huggingface/tokenizers/tree/f383101a26663708484cac0727792aad74f78234), whose complete [Apache-2.0 license](docs/licenses/Tokenizers-Apache-2.0.txt) is preserved. Neither native tokenizers nor Transformers is a product dependency.

The generated Unicode compatibility data records observed scalar classification, normalization and lowercase behavior of that reference backend. Prompt-weight parsing separately records factual Python 3.12 Unicode 15.0.0 decimal-digit and whitespace ranges; no Python implementation source is copied for those tables. Applicable source/data notices are retained alongside this provenance:

- **Unicode, Inc., copyright 1991-2026**, [Unicode License V3](docs/licenses/Unicode-3.0.txt), copied from the [official license](https://www.unicode.org/license.txt) on 11 September 2026 (SHA-256 `e7a93b009565cfce55919a381437ac4db883e9da2126fa28b91d12732bc53d96`).
- **The Rust Project Developers**, [Rust MIT notice](docs/licenses/Rust-MIT.txt), copied from [the identified compiler source](https://github.com/rust-lang/rust/blob/ded5c06cf21d2b93bffd5d884aa6e96934ee4234/LICENSE-MIT). Rust's lowercase tables identify Unicode 17.0.0.
- **The Rust Project Developers, copyright 2015**, [unicode-normalization-alignments MIT notice](docs/licenses/unicode-normalization-alignments-MIT.txt), preserved from the [published 0.1.12 crate](https://static.crates.io/crates/unicode-normalization-alignments/unicode-normalization-alignments-0.1.12.crate). Its generated table header also credits **Copyright 2012-2018 The Rust Project Developers** and declares Unicode 9.0.0. The crate is offered under MIT or Apache-2.0; the MIT notice is retained here.
- **K.Kosako, copyright 2002-2021**, [Oniguruma BSD-style notice](docs/licenses/Oniguruma-BSD.txt), preserved from `oniguruma/COPYING` in the [published onig_sys 69.9.1 crate](https://static.crates.io/crates/onig_sys/onig_sys-69.9.1.crate), whose bundled history identifies the Unicode 16.0 update.

These notices preserve the reference components' attribution. The generated data and C# implementation do not distribute the Python/Rust laboratory binaries. Binary packaging still requires the separate complete dependency notice bundle and SBOM described above.

The optional [isolated CLIP source laboratory](labs/clip-source/README.md) executes selected, hash-verified ComfyUI GPLv3 source definitions in a separate environment. Its Python/PyTorch/NumPy/einops dependencies are pinned for diagnostic collection only and are not included in the application or .NET test distributions. Its synthetic tensor artifacts do not contain pretrained model weights or native dependency binaries.

## CPython Unicode lowercase

`src/ComfySharp.Nodes/PythonUnicodeLower.cs` and its Unicode 15.0.0 resource derive full lowercase mappings, Cased / Case_Ignorable flags and Final_Sigma behavior from CPython **3.12.10**, tag resolved to commit [`0cc81280367df838c4b199f8f0378837165071c2`](https://github.com/python/cpython/tree/0cc81280367df838c4b199f8f0378837165071c2). The exact sources are `Objects/unicodetype_db.h`, `Objects/unicodectype.c`, `Objects/unicodeobject.c` and the version declaration in `Tools/unicode/makeunicodedata.py`. Source hashes, static extraction and the C# modifications are documented in [the table provenance](labs/python-unicode-tables/README.md).

The complete [CPython 3.12.10 license](docs/licenses/CPython-3.12.10.txt) is copied byte-for-byte from [LICENSE at that commit](https://github.com/python/cpython/blob/0cc81280367df838c4b199f8f0378837165071c2/LICENSE), SHA-256 `3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf`, preserving the Python Software Foundation and historical copyright notices and conditions. The Unicode character helpers credit Marc-Andre Lemburg and Fredrik Lundh and copyright Corporation for National Research Initiatives. Unicode data attribution and permission are retained in [Unicode License V3](docs/licenses/Unicode-3.0.txt), identified above. The adaptation extracts static lowercase/property data and implements scalar validation, linear contextual-sigma passes and cancellation; it does not include a Python interpreter or claim to implement unrelated casing operations.

`PythonUnicodeCase.cs` additionally adapts full upper/title data and the Capitalize/Title control flow from those same pinned CPython sources. Its separate [upper/title table provenance](labs/python-unicode-case-tables/README.md) describes the changes: static scalar mappings, immutable C# dictionaries, strict validation and cancellation, and reuse of original-text context with distinct immediate-Cased and Final_Sigma states. The existing CPython and Unicode notices above remain applicable. Lower's historical table is unchanged; no casefold/swapcase tables or Python runtime are distributed by this addition.

`LoraModelAliases` and `LoraNodes` port the local SD/CLIP LoRA naming and node contracts from ComfyUI `comfy/lora.py` and `nodes.py` at commit `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a` (GPLv3-or-later, Copyright Comfy). The source alias collector also executes the frozen `comfy/utils.py` naming map. Original source notices remain applicable.
