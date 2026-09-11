# Third-party provenance

ComfySharp is an independent C# port of ComfyUI. It is not an official Comfy-Org release.

Functional reference: [ComfyUI at 1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a), copyright ComfyUI contributors, GNU GPL version 3. The complete upstream GPL text is preserved in LICENSE.

Editor reference: [ComfyUI_frontend v1.51.10 at e7d1c7fc6823e330fdab524610b0000394cb1dbc](https://github.com/Comfy-Org/ComfyUI_frontend/tree/e7d1c7fc6823e330fdab524610b0000394cb1dbc). C# implementations cite their corresponding reference files. No compiled upstream frontend is distributed.

The sigma generators in `src/ComfySharp.Inference/SigmaSchedules.cs` are ported from the frozen ComfyUI `comfy/k_diffusion/sampling.py`, with numeric references recorded in `docs/SIGMA_SCHEDULES.md`. The original k-diffusion project's notice, copyright (c) 2022 Katherine Crowson, is retained in [k-diffusion-MIT.txt](docs/licenses/k-diffusion-MIT.txt), copied from [this pinned licence source](https://github.com/crowsonkb/k-diffusion/blob/4601bf085320592473f681a62808ed873d17fad5/LICENSE). This notice is preserved alongside the ComfyUI provenance and project GPLv3 licence; it does not change the frozen functional reference.

## PreviewAny and tensor text

`src/ComfySharp.Inference/TensorPreviewFormatter.cs` adapts the dense real-number formatting logic in PyTorch's [`torch/_tensor_str.py` at cf30153c4c131c8164ee7798e5022d810682e2cb](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/_tensor_str.py): numeric width and notation, scalar/vector/multidimensional layout, summarized edges and suffix placement. The complete PyTorch copyright statements, redistribution conditions and disclaimer are retained in [PyTorch-BSD-style.txt](docs/licenses/PyTorch-BSD-style.txt), copied from [LICENSE at the same commit](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/LICENSE). The C# adaptation has an explicit supported representation scope; it does not include the Python module or claim to port every PyTorch tensor representation.

The isolated fixture laboratory reports `torch.version.git_version = cf30153c4c131c8164ee7798e5022d810682e2cb` and `torch.__version__ = 2.13.0+cu130`. Its installed `_tensor_str.py` was compared with the official file at that commit and is identical after CRLF-to-LF normalization (normalized UTF-8 SHA-256 `5fcb2c4e80feb8f7d458074a808c00cf0fdc2a67b964a11e7b5ece7bdd135d87`). The CPU output references are preserved in [preview-any.cpu.json](tests/ComfySharp.Inference.Tests/Fixtures/preview-any.cpu.json). This provenance identifies the formatting reference and laboratory only; ComfySharp's libtorch runtime remains the separately locked 2.10 dependency, and the distributed tests do not execute Python.

The `PreviewAny` node in `src/ComfySharp.Nodes.Tensor/TensorNodes.cs` follows [ComfyUI's frozen `comfy_extras/nodes_preview_any.py`](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_preview_any.py), under the project GPLv3 provenance above. That file explicitly credits the original Display Any implementation in **rgthree/rgthree-comfy**. We preserve that attribution to **Regis Gaughan, III (rgthree), copyright (c) 2023**, and its complete MIT notice in [rgthree-comfy-MIT.txt](docs/licenses/rgthree-comfy-MIT.txt). The identified upstream [Display Any source](https://github.com/rgthree/rgthree-comfy/blob/5288408220180af41ce50b0d29135e1ef5f83fdb/py/display_any.py) and [MIT licence](https://github.com/rgthree/rgthree-comfy/blob/5288408220180af41ce50b0d29135e1ef5f83fdb/LICENSE) are pinned at `5288408220180af41ce50b0d29135e1ef5f83fdb` to make this credit verifiable; ComfyUI's frozen node remains the functional reference. No third-party Python or JavaScript extension is executed by this port.

## Distribution dependencies

Dependencies retain their own licences: .NET/ASP.NET Core, Avalonia, Nodify.Avalonia and TorchSharp (MIT); libtorch (BSD-style, with additional bundled notices); SQLite (public domain), Microsoft.Data.Sqlite (MIT), SQLitePCLRaw (Apache-2.0); Skia/SkiaSharp (BSD/MIT notices). Exact transitive packages are recorded in packages.lock.json. Packaging must copy every applicable native and managed dependency notice and produce an SBOM before release.

FFmpeg and ANGLE are planned dependencies, not bundled by this bootstrap. Their build configuration, licence obligations and notices must be recorded when integrated. Model weights and tokenizer resources are not included; each future compatibility record must identify its licence, origin and SHA-256.

Binary packaging is currently local validation only: dependency notice collection
and the exact distribution SBOM are incomplete, so the portable workflow does
not upload binaries. This summary is not a complete distribution notice bundle.
See [packaging requirements](docs/packaging.md) for the unresolved release gate.
