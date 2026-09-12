# LoRA optimizers and losses

`LoraTrainingOptimizer` implements the frozen training defaults of Adam, AdamW,
SGD and RMSprop with C# orchestration and native tensor operations. It retains
ordinary Float32 `TrainableLoraPatch` parameters and owns their gradient buffers
and optimizer state. Use one optimizer per set of factors; serialize all factor,
model, backward, update and export operations.

Adam and AdamW use betas `(0.9, 0.999)` and epsilon `1e-8`. AdamW uses decoupled
weight decay **0.01**, the default inherited by ComfyUI from PyTorch. SGD has no
momentum or decay. RMSprop uses alpha `0.99`, epsilon `1e-8`, no momentum, no
centering and no decay. These options are explicit source contracts, not the
defaults of another binding. Adam variants need two additional factor-sized
state tensors, RMSprop one and SGD none; activations remain additional memory.

`TrainingLoss.Calculate` provides MSE, L1, Huber and SmoothL1, with mean reduction
and delta/beta 1. It converts supported floating predictions and matching targets
to Float32 while retaining autograd. This input conversion does not qualify mixed
precision model training or GradScaler.

For each microbatch, compute a differentiable scalar loss and call `Accumulate`.
It divides the loss by `AccumulationSteps` before backward. Call `Step` only after
a complete window. Previously inactive and newly inactive parameters have null
gradients and are skipped, including decay and per-parameter state updates.
`ResetAccumulation`, cancellation and rejected gradients discard the unfinished
window. Prior successful updates remain. A native update error or nonfinite
updated factor faults the session; callers must discard it, as updates cannot be
rolled back atomically. Cancellation is observed between native operations.

The [source collector](../labs/lora-training-source/optimizers.py) executes the
frozen `_create_optimizer` and `_create_loss_function` factories. Its 16 combinations
cover 96 microbatches, 48 updates, changing parameter participation and four loss
boundary gradients. Comparison bounds were fixed at `2e-6` absolute plus `2e-6`
relative before evaluation. Unit tests also exercise cancellation, retained
ownership, invalid gradients, update overflow and deterministic release.

The existing `sd-lora-train` diagnostic additionally accepts `--optimizer`, `--loss`
and `--accumulation-steps`. Defaults remain SGD/MSE/1. It uses supplied SD1.5
weights directly and exports a new small adapter; it never downloads or copies
a checkpoint. Its miniature inputs remain synthetic; the [denoised-latent objective](LORA_DENOISING.md)
can now be selected explicitly. The [dataset loop](TRAINING_DATASETS.md) supplies
the plain Float32 batch/RNG profile. Full training nodes, checkpointing, offload and mixed precision
remain required; no complete training node or model-family qualification follows.

The [campaign record](qualification/lora-optimizers.json) identifies the source
corpus, tests, checkpoint, adapter, hardware and limits of the CUDA diagnostic.
