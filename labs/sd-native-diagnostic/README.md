# Isolated native-origin control

This experiment holds the C# assemblies, input recipe, exact source scripts and
numerical profile fixed while changing only the native-library origin requested
for fresh product processes. Its separate Ubuntu workflow runs six processes on
one host: source, original NuGet product and wheel-native product, each under
automatic and DEFAULT ATen dispatch. It does not rerun earlier diagnostic workflows.

The source and comparison helpers are imported only after checking their pinned
SHA256s. The source runner is unchanged. Dependencies and frozen source blobs keep
their existing hashes. Product tests use the original build with `--no-build` and
`--no-restore`; all original build files and accepted fixtures are hashed before
and after the experiment. No source, native binary, assembly or reference is copied
over the original build.

For the wheel-native child only, absolute paths to the wheel's `libc10.so`,
`libtorch_cpu.so`, and `libtorch.so` are supplied through `LD_PRELOAD`, in that order,
with `LD_LIBRARY_PATH` selecting the wheel's library directory. Neither Python nor
`libtorch_python` is preloaded. Existing nonempty loader overrides make the
experiment invalid; they are not silently removed or merged. `PATH` and all
unrelated environment values remain unchanged. Evidence stores basenames and
hashes rather than private paths or memory addresses.

ELF SONAME and DT_NEEDED entries are recorded before execution. Environment intent
is insufficient: the product's actual `Process.Modules` records must contain
exactly one of each requested core library, with the selected hashes, and the
original `libLibTorchSharp.so` hash. The wheel variant must load wheel OpenMP and
must contain no second NuGet core, no NuGet OpenMP, and no Python binding. An
absolute .NET fallback load or loader incompatibility can defeat substitution;
that makes the variant invalid and prevents attribution of a numerical comparison
to the requested origin. There is no resolver workaround or silent fallback.

Each existing SD15 square test records its 43 fine tensors, parameters/input
identities, layouts, and observer off/on/off hashes before the committed-reference
assertion. The collector checks that exactly this one test ran. Nonzero test or
process status remains nonzero. It reports source/source, same-mode product/source,
and product/product comparisons only after loaded-origin validation. A successful
diagnostic or closer result does not adopt wheel libraries into the product, alter
acceptance, or qualify another shape, model family, backend, or platform.
