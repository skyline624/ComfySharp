# SD native package comparison on independent build copies

The [isolated Linux diagnostic](https://github.com/skyline624/ComfySharp/actions/runs/34645018348)
at commit `3583d61a2cef113bf0e1a49fca0b95c704b5c7a8` completed eight fresh
processes: source, original product, an unchanged NuGet build copy and a copy
using the source wheel's native core/OpenMP libraries, each in automatic and
explicit DEFAULT modes. All processes exited successfully. All six existing
product tests passed. This is a controlled diagnostic of one reduced SD15 square
case, without pretrained weights or new model compatibility.

## Integrity and execution origins

The host reports Intel Xeon Platinum 8573C; source reports AVX512 in automatic
mode and DEFAULT when requested. Intra/inter-op threads are 1/1. Each product
process records one c10, torch_cpu, torch, unchanged TorchSharp bridge and GNU
OpenMP image with the intended identities. There are no duplicate relevant core
images, mixed OpenMP origins or loaded Python bindings in those product maps.
The earlier duplicate-registration failure is absent here. Source library lists
are wheel file inventories, not actual loaded-module maps.

Before/after inventories match for the original build, both copies, source
snapshot, accepted fixtures and wheel input. The control copy has exactly the
original 121 files. The wheel copy retains 117 files unchanged, replaces the
three core assets and OpenMP dependency, and adds the OpenMP SONAME alias. The
aliases share an internal inode and remain independent of the package input.
Managed code, dependency/runtime configuration, test adapters and the native
bridge remain identical. Large native files were not uploaded; their inventory
and inode checks are attestations from the pinned producer. Actual product
module hashes connect those identities to the running processes.

Two independent audits decoded and rehashed all 368 tensor/input records.
The first performed 473 tensor comparisons, the second 344 overlapping
comparisons. All 686 parameter identities agree across processes. Thirteen
script/dependency files match their canonical committed blobs; three frozen
upstream blobs and selected AST hashes were independently reconstructed.
All three output repetitions in each process are bit-identical.

## Numerical result

| Comparison to source in the same mode | Exact captures | Final maximum absolute error | Values outside the fixed profile |
|---|---:|---:|---:|
| Automatic, original NuGet | 16/43 | 0.00001239776611328125 | 0 |
| Automatic, NuGet copy | 16/43 | 0.00001239776611328125 | 0 |
| Automatic, wheel-native copy | 43/43 | 0 | 0 |
| DEFAULT, each of the three product variants | 43/43 | 0 | 0 |

The original and unchanged NuGet copy also match exactly in both modes,
including captured input bytes, shapes, strides, alignment flags and parameter
layouts. The copy/invocation control therefore passes before interpretation of
the native package contrast.

In automatic mode, the first differing source-ordered capture remains
`input_blocks.10.0.in_layers.0.output`: six of 128 GroupNorm values differ by at
most `1.1920928955078125e-7`. The captured GroupNorm input, gamma and beta agree,
including their layouts and alignment flags. All earlier captured outputs agree.
The wheel-native product reproduces this boundary and every later capture
exactly. The fixed `3e-5 + 3e-5 * abs(expected)` profile is unchanged.

Borrowed latent/timestep buffers are not aligned identically in every process:
the original/control report unaligned buffers in automatic mode, while the
wheel copy reports alignment. Context is unaligned in all three products.
These differences remain in the evidence; exact intermediate input captures at
the first differing GroupNorm do not prove that every unobserved allocation or
backend state is controlled. The audit orders product/product comparisons using
the source capture sequence: the producer's dictionary begins with `output`,
so its first-difference label alone is not a temporal localization.

The result supports a native-package difference on this case and host. Core
libraries and OpenMP change together, so it does not isolate a specific binary,
compiler flag or kernel. It does not justify adopting the wheel libraries in
the product or close earlier Windows/Linux failures. Wider U-Net/CFG, VAE,
sampling, resource and platform checks remain necessary.

[Machine-readable observations and independent comparisons](sd-native-copy-3583d61.json)
retain the library identities, inventories, per-capture metrics and artifact
metadata. Archive digests are GitHub-reported identities, not locally rehashed
ZIP files. The [normal CI campaigns](sd-cpu-variation-4b63561.md) remain separate.
