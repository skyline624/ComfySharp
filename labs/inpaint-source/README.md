# Frozen inpaint references

`reference.py` is a development-only collector. Run with PyTorch 2.10.0+cpu in a
separate reference laboratory, passing `--source` for a Git checkout containing
backend commit `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`, and `--output` for a
new JSON file. The collector refuses to overwrite an existing reference.

It extracts and executes the actual `reshape_mask`, `repeat_to_batch_size`,
`EPS`, `reshape_sigma`, `KSamplerX0Inpaint`, `VAEEncodeForInpaint` and
`SetLatentNoiseMask` declarations with Git/AST. It records the source blob hashes.
No ComfyUI runtime imports, model downloads, package installations or C# outputs
participate in reference generation. `SetLatentNoiseMask` is extracted for
provenance; its map-retention behavior is tested separately in .NET.

Two resize cases cover batch repetition/truncation and fractional values; three
mask cases each cover three sigma vectors; five image cases cover growth
0/1/2/6/64, nonmultiple-of-eight dimensions, a shared mask, two RGBA images and
round-to-even values. The model is explicitly analytical and the VAE only captures
its input. These cases establish transformation contracts, not pretrained parity.

Absolute and relative bounds of `1e-6` were fixed before comparison. The .NET test
suite embeds the resulting JSON and never runs Python. The original procedural
`inpaint-input.png` fixture contains a colored disk, a gradient background and a
central transparent region with a partially transparent border. It is a workflow
input, not a generated-image quality reference; it contains no external artwork.
