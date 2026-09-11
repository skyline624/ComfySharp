# Prospective reduced SD1.5 pipeline reference

This directory currently publishes the [frozen protocol](protocol.json), before
any pipeline forward. It specifies four synthetic cases linking source CLIP
conditioning, EPS/CFG, Euler and classical VAE decoding. The
[comparison profile](../../docs/qualification/sd15-pipeline-native210-cpu-f32-v1.md)
describes inputs, operation order, exact sigma checks and numerical bounds.

No pipeline outputs are included. The source collector will be reviewed and
published separately before its first numerical execution. Its references must
come from the frozen ComfyUI source, independently of the C# results. Python is
reserved for this separate laboratory and is not an application dependency.

These reduced cases do not qualify a pretrained checkpoint, stock dimensions,
complete ComfyUI workflows or a model family.
