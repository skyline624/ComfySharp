# Prospective uniformly aligned reduced native suite

This is a new diagnostic protocol; it never updates the 6b1347c collection or its laboratory.
The unchanged fourteen reference tests keep their expected outputs, tolerance and three assertions.
The dedicated COMFYSHARP_SD_ALIGNED_SUITE_TRACE_DIR option selects a separate test helper.
All earlier observation/offset options are incompatible. The ordinary path is unchanged when absent.

Source and product prepare every actual input per case, in the protocol order, with unconditional
native empty/copy operations. Each copy is CPU/F32, independent, contiguous and zero-offset;
its data address must be divisible by 64. Shapes/strides/bytes must equal the borrowed inputs.
Inputs are reused for three forwards and inspected immediately before/after each call.
No address is serialized; only modulo64 is recorded. Unexpected layouts fail; there is no retry.

The new source orchestration reuses the unchanged source model, CFG primitives and audited
denoise/separate function ASTs. It intentionally gives each CFG policy its own prepared inputs
and computes concatenated contexts within each call, matching the product policy boundary.
Source and product observe U-Net off/on/off; CFG remains off/off/off. These allocation/timing
changes are explicit and may change numerical output. They do not establish normal-CI behavior.

Eight processes on one Linux host remain source/original/NuGet-copy/wheel-copy x auto/default.
Strict native maps, ELF closure, unchanged binding/managed assemblies, hardlink aliases,
inventories, raw lock checks, parameter/input hashes, TRX/exit coherence and repeats remain.
Original/copy must match every capture exactly with equal layouts/offset/alignment residues;
wheel/NuGet input and parameter gates must also pass before bundle interpretation.
Missing/nonzero allocation metadata invalidates eligibility, even with exact output bytes.
The actual inputs are recorded rather than an unused normalized buffer.

Accepted-test failures and same-host diagnostic agreement are separate. No fixture replacement,
tolerance change, fallback, primitive attribution, pretrained or family qualification is allowed.
Source generators, old laboratories, product code and package locks are unchanged.
Native package trees are excluded from artifacts as in the parent copy diagnostic.
