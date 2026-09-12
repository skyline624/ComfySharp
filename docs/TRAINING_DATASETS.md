# SD training datasets, batches and random sequence

`SdTrainingDataset` snapshots supplied plain SD Float32 latents into owned CPU
storage. A single input keeps its batch. Multiple inputs split into image rows,
then concatenate if their shapes agree; otherwise they remain multiresolution.
Explicit bucket mode preserves the supplied batches as separate groups. Snapshots
and the full initial guide-noise allocation each have a default 512 MiB allowance;
temporary conversion/concatenation buffers, contexts and activations are additional.

The source's latent-format boundary is preserved explicitly:

| Mode | Data sent to the objective |
|---|---|
| Standard | Latent-format scaling, default SD1.5 factor 0.18215; zero input remains unchanged |
| Multiresolution | Separate original dataset rows, without the guider's scaling |
| Buckets | Separate original buckets, without the guider's scaling |

This difference comes from the frozen `TrainSampler` reading `real_dataset` or
`bucket_latents` while the guider scales only its own `latent_image`. It is not
silently normalized by the port. Other latent formats and changes to this behavior
require explicit source qualification.

`SdTrainingBatchSampler` owns a private CPU generator. It consumes the initial
full guide noise, selects images using `randperm` (or weighted `multinomial` then
`randperm` for buckets), resets to `seed + microbatch * 1000` for each noise call,
then draws each sigma percentage individually. Multiresolution resets again for
each selected image, exactly as `prepare_noise` does. This reproduces the source's
global-generator sequence without changing the application's global CPU RNG.
It assumes the implemented plain model forward does not consume that generator;
additional stochastic model operations need their own integration.

Returned batches own their latent/noise/sigma tensors; global image indices select
the corresponding captions. Multiresolution batches contain one group per image.
The sampler retains the dataset, so disposing its caller's reference is safe.
Cancellation before sampling consumes no state. An error after sampling begins
faults the sampler; no partial batch is returned. Seeds never silently wrap.

`SdLoraTrainingLoop` joins these batches to the denoising objective and optimizer,
with one plain whole-image text context per image. A shared caption expands outside
bucket mode. Multiresolution loss follows source order `sum / grad_acc / image_count`;
its callback also reports this normalized loss, while standard/bucket callbacks
report raw loss. Cancellation clears unfinished accumulation. Previously completed
updates are retained; callers discard a cancelled job's adapters as appropriate.

The [source collector](../labs/lora-training-source/batches.py) executes the actual
frozen preparation, initial-noise and batch-selection methods. Six cases and 24
microbatches compare selected indices, latent/noise/sigma values and native RNG
state hashes exactly on the local Windows CPU runtime. It includes high unsigned
seeds, oversized batches, mixed resolutions, buckets and empty-valued latents.
Model evaluation is replaced by a recording loss in this corpus: it does not
qualify a complete pretrained training workflow.

`sd-lora-train --objective denoised-latent --dataset-mode standard|multi-resolution|buckets`
runs an opt-in real-checkpoint diagnostic over three miniature synthetic examples.
It performs two optimizer updates, exports a small adapter and verifies reload
equality. It reads the shared checkpoint directly without copying or downloading
models. The [campaign record](qualification/training-batches.json) identifies the
hardware, tests and artifacts. Full node integration, adapter initialization and
all targets, rich conditioning, image datasets/style quality, other dtypes and
architectures, checkpointing/offload and required platforms remain pending.
