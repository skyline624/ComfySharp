# Differentiable SD LoRA and export

`SdUnet.ForwardForTraining` runs the implemented plain SD U-Net with autograd
enabled. Ordinary `Forward` remains inference-only. Base parameters stay frozen;
only selected weights are calculated from caller-owned `TrainableLoraPatch` leaves.
Unchanged native weight storage is shared. A returned prediction keeps its saved
native graph alive even if the original model is subsequently disposed.

Each trainable patch snapshots two Float32 matrices as independent leaf parameters,
with explicit alpha. Conv factors use a flattened input/kernel dimension. Callers
must serialize forward, backward, parameter updates and export; public parameter
wrappers are borrowed and must not be disposed, resized or moved. `Snapshot`
produces an independent frozen patch for ordinary inference. LoCon/DoRA training,
checkpointing, offload and quantized backward remain required. The
[two-factor SD Float32 training bypass](LORA_TRAINING_BYPASS.md) now preserves alpha
gradients and is connected to the denoiser and dataset loop through `BypassMode`.

`LoraTrainingFile.SaveNew` writes explicit alias prefixes with `lora_up.weight`,
`lora_down.weight` and scalar `alpha`. CPU factor snapshots are Float32; alpha is
Float64 to preserve the API's value. A sibling temporary file is flushed and moved
without overwriting an existing destination. Validation or cancellation does not
publish a partial destination. This is an adapter writer, not a general checkpoint
serializer. The factor snapshot and differentiable weight allowances default to
512 MiB each; saved autograd activations and conversion buffers are additional.

The [source laboratory](../labs/lora-training-source/README.md) compares raw
predictions, loss, gradients and two SGD updates using the actual frozen U-Net
and `LoraDiff`. First and final convolution targets exercise backward across the
full reduced SD1/SD2 topology. Bounds of `3e-5` absolute plus `3e-5` relative were
fixed before comparison. Tests also cover cancelled graphs, ownership, frozen base
weights and export/reload through the normal LoRA file loader.

The opt-in .NET diagnostic `sd-lora-train` loads a supplied real SD1.5 checkpoint,
runs two optimizer steps on miniature synthetic inputs, exports a small adapter
and checks reloaded prediction equality. It defaults to SGD/MSE/raw predictions;
the [denoised-latent objective](LORA_DENOISING.md) is selectable explicitly.
It needs existing output directories:

```text
dotnet run --project tools/ComfySharp.RuntimeProbe -c Release -- sd-lora-train --checkpoint MODEL.safetensors --adapter-output NEW.safetensors --report NEW.json --device cpu
```

For the existing Windows CUDA build, use `-p:NativeBackend=cuda` and `--device cuda:0`.
No checkpoint is copied or downloaded. The [optimizer and loss implementation](LORA_OPTIMIZERS.md)
now supplies Adam, AdamW, SGD, RMSprop and accumulated gradients. Optional diagnostic
flags select these; the original defaults remain SGD/MSE/1. This is not dataset training,
the source denoising/noise schedule or a style-quality assessment.
`TrainLoraNode`, `LoraModelLoader`, `SaveLoRA` and `LossGraphNode` are
not announced complete by these APIs. The catalogue stays at 48 registered nodes.
Full training, other architectures/dtypes, source comparison of pretrained training,
and platform qualification remain mandatory. See the [campaign record](qualification/lora-training.json).
