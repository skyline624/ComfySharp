# SD1.5 checkpoint assembly: three-platform CI observation

The 25 reduced synthetic checkpoint assembly tests pass on all three platforms
in [Build and test run 34653895731](https://github.com/skyline624/ComfySharp/actions/runs/34653895731),
at commit [`47e3e2996953e0049e58f8ba345936e11865955e`](https://github.com/skyline624/ComfySharp/commit/47e3e2996953e0049e58f8ba345936e11865955e).
The overall run fails because the existing Linux numerical reference failures
and macOS CLIP stock failures remain. The [machine-readable record](sd15-checkpoint-ci-47e3e29.json)
contains the artifact inventory, TRX hashes, step outcomes and runtime observations.

## Results and counting

All three Release builds complete with zero warnings and zero errors. The audit
recounts **18 artifacts and 41 TRX reports**, including one new checkpoint report
per platform.

| Platform | Checkpoint contracts | First-use reference cases | Unique ordinary + stock tests | Job |
|---|---:|---:|---:|---|
| Windows x64 | 25 passed | 190 passed | 1,322 passed | Success |
| Linux x64 | 25 passed | 175 passed, 15 failed | Not executed | Failure |
| macOS arm64 | 25 passed | 190 passed | 1,320 passed, 2 failed | Failure |

The checkpoint cases comprise nine metadata/admission cases and sixteen native
assembly cases. Their names match the [assembly evidence](sd15-checkpoint-assembly.json).
They are repeated in the aggregate suite when that suite is reached; they do
**not** add 25 to its 1,322 unique test IDs. The 190 first-use cases similarly
repeat existing tests in separate processes. Linux has 215 distinct observed
test IDs across checkpoint and first-use processes; its absent aggregate suite
is not counted as passed or ignored. There are no ignored cases among the tests
that actually executed.

The checkpoint step precedes the numerical reference steps, so its results are
available even where later comparisons fail. Its process is separate from each
first-use reference process. No assertion, failure policy or numerical tolerance
was relaxed.

## Existing numerical results

Linux retains two U-Net, six CFG and seven Euler first-use failures. Its aggregate
suite, catalogue check, metadata/tokenizer/CLIP command-line checks, Desktop smoke,
native operation probe and full CLIP stock comparisons are not executed after
that failed step. The attempted upload of absent Linux stock traces also fails;
it is an artifact-collection consequence, not another numerical test failure.
macOS completes the aggregate and subsequent smoke checks, then fails the two
full CLIP L/G stock reference cases.

All **80 U-Net trace payloads per platform** and **81 Euler comparison payloads
per platform** are identical to [the preceding Euler CI observation](sd-euler-ci-2087602.md).
That record remains the detailed numerical reference for these unchanged results:
Windows and macOS have eight passing Euler cases and 81 bit-exact comparisons;
Linux has one passing case, 27 bit-exact comparisons out of 81 and 260 distinct
final output elements outside the fixed bound. Counting every observation,
including three repeated final outputs, gives 1,587 outside-bound elements.
The 31 input records and 686 parameters per case before/after remain exact;
each case's three outputs remain identical to one another. The nine Euler
behavior tests pass in the Windows/macOS aggregate suites and are not executed
on Linux in this run. No new numerical cause is inferred from this repetition.

One small change occurs in the Windows CLIP-G diagnostic capture: the actual
`Forward` result `ProjectedPooled` differs from run 34651839113 in **2,029 of
2,560 Float32 values**, with maximum absolute difference
**4.76837158203125e-7**. Its SHA-256 changes from
`db6799d5e41bf0f43dc75d9c333a58a3ca53fb5b838a011f1a5407a65fff5e3b` to
`a752541ba423c6fdee686e115595123922b02693c995185a794716a74312198e`.
This is a real graph output, not a reconstructed first-layer Q/K/V diagnostic.
The other eight G records and all nine L records are unchanged. All three
Windows stock tests still pass. The difference quoted here is against the
preceding CI capture; it does not establish a primitive or allocation cause.
macOS's 18 CLIP trace payloads are unchanged.

## Scope and limits

The new tests validate the same reduced synthetic loading, admission, rollback
and independent ownership contracts described in the
[assembly contract](sd15-checkpoint-assembly.md). They execute CLIP
pooling/projection and acquire/retain U-Net and VAE factories; they do not run a
composed generation pipeline or load a stock pretrained checkpoint. The separate
metadata-isolation harness remains a local Windows observation, not a new
three-platform isolation gate.

The runtime identities in the JSON come from traced U-Net processes, not the
checkpoint tests. Windows and Linux report AMD EPYC 7763, one intra-op and one
inter-op thread, available AVX2 and unavailable AVX-512 intrinsics. Actual ATen
dispatch remains unobserved; CPUID and .NET intrinsic availability do not measure
it. Loaded native library hashes are recorded on Windows/Linux; the macOS module
observation is explicitly unavailable. These are separate CI hosts and processes,
not a controlled same-host source/product experiment.

Since commit 2087602, the source/test changes are the checkpoint loader, contracts,
owner, three supporting test files and the safetensors reader-state guard.
Existing graph math, numerical reference tests, fixtures and package versions
are unchanged. No pretrained weights, successful stock checkpoint load,
complete generative workflow or model family is qualified by this run.
