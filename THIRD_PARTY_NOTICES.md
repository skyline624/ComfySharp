# Third-party provenance

ComfySharp is an independent C# port of ComfyUI. It is not an official Comfy-Org release.

Functional reference: [ComfyUI at 1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a), copyright ComfyUI contributors, GNU GPL version 3. The complete upstream GPL text is preserved in LICENSE.

Editor reference: [ComfyUI_frontend v1.51.10 at e7d1c7fc6823e330fdab524610b0000394cb1dbc](https://github.com/Comfy-Org/ComfyUI_frontend/tree/e7d1c7fc6823e330fdab524610b0000394cb1dbc). C# implementations cite their corresponding reference files. No compiled upstream frontend is distributed.

The sigma generators in `src/ComfySharp.Inference/SigmaSchedules.cs` are ported from the frozen ComfyUI `comfy/k_diffusion/sampling.py`, with numeric references recorded in `docs/SIGMA_SCHEDULES.md`. The original k-diffusion project's notice, copyright (c) 2022 Katherine Crowson, is retained in [k-diffusion-MIT.txt](docs/licenses/k-diffusion-MIT.txt), copied from [this pinned licence source](https://github.com/crowsonkb/k-diffusion/blob/4601bf085320592473f681a62808ed873d17fad5/LICENSE). This notice is preserved alongside the ComfyUI provenance and project GPLv3 licence; it does not change the frozen functional reference.

Dependencies retain their own licences: .NET/ASP.NET Core, Avalonia, Nodify.Avalonia and TorchSharp (MIT); libtorch (BSD-style, with additional bundled notices); SQLite (public domain), Microsoft.Data.Sqlite (MIT), SQLitePCLRaw (Apache-2.0); Skia/SkiaSharp (BSD/MIT notices). Exact transitive packages are recorded in packages.lock.json. Packaging must copy every applicable native and managed dependency notice and produce an SBOM before release.

FFmpeg and ANGLE are planned dependencies, not bundled by this bootstrap. Their build configuration, licence obligations and notices must be recorded when integrated. Model weights and tokenizer resources are not included; each future compatibility record must identify its licence, origin and SHA-256.

Binary packaging is currently local validation only: dependency notice collection
and the exact distribution SBOM are incomplete, so the portable workflow does
not upload binaries. This summary is not a complete distribution notice bundle.
See [packaging requirements](docs/packaging.md) for the unresolved release gate.
