# Isolated square U-Net operator diagnostic

This laboratory investigates the first measured SD15 square-case difference
between the frozen source and product. Its own push-triggered Linux workflow runs
four fresh processes on one host: source and product under automatic ATen dispatch,
then source and product under `ATEN_CPU_CAPABILITY=default`. It records the actual
automatic source capability. It neither assumes AVX512 nor repeats the earlier
31-process dispatch/allocation experiment.

The source imports only hash-verified, unchanged laboratory helpers and their exact
frozen U-Net ASTs. The three source blobs and CPU dependencies retain their existing
pins. Parameters and inputs use the same deterministic recipe and full reduced
SD15 topology. No accepted expected output is read to compute source results.

Forty-three tensors cover timeEmbedding, down2, the stride-2 convolution at
`input_blocks.9.0.op`, and every GroupNorm/SiLU/convolution/time-linear input and
output in residual blocks 10 and 11, their block inputs/time embeddings/final sums,
and the final model output. The actual input to `out_layers.0` captures the time
addition; its arithmetic is never reconstructed in the diagnostic. Module hooks
observe exact source operations. Shape, stride, alignment modulo 64 (boolean),
Float32 bytes and hashes are preserved without publishing memory addresses.

Each process runs observer-off, observer-on, observer-off on the same model and
inputs, and must produce the same final output hash three times. The product uses
the existing square reference test with an opt-in observer, and performs its
unchanged committed-output comparison after writing the trace. Missing or changed
captures, parameter/input identities, or observer neutrality produce an integrity
failure. Test exit statuses and logs remain available even when comparison fails.

The collector reports source/source and all four product/source pairings separately.
It reports the first differing record in source execution order, even if that record
is timeEmbedding or down2: another runner may differ before the selected interval.
A primitive is not identified as the cause merely because its output differs;
its incoming bytes and parameter/layout evidence must also be checked.
Numerical comparisons use the existing fixed bound for diagnosis; closer results
cannot replace committed references or grant qualification. Source/parameter
hashes, runtime build identities, CPU flags, and loaded/built native-library hashes
remain evidence, not interchangeable package-version claims.

Outputs must be new absolute directories outside the repository and source
snapshot. Neither script downloads models, writes fixtures, changes native dispatch
inside an already initialized process, or modifies the six accepted source scripts.
This folder's workflow does not alter or trigger the earlier runtime workflow.
