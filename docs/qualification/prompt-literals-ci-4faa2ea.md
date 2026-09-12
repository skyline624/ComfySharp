# Prompt literals and case-sensitive HTTP dictionaries — CI 4faa2ea

The 33 new application regression cases passed on Windows, Linux and macOS: **99 successful case executions**, with no failed or skipped case in this selection. The [CI run 34663882926](https://github.com/skyline624/ComfySharp/actions/runs/34663882926) nevertheless **failed globally**. Its exact commit is `4faa2eac5a87ac67e575d61922c1b3ef4c396a1d`.

| Evidence | Windows x64 | Linux x64 | macOS arm64 |
|---|---:|---:|---:|
| New Workflow literal cases | 18 passed | 18 passed | 18 passed |
| New Host literal cases | 8 passed | 8 passed | 8 passed |
| New Host case-sensitive dictionary cases | 7 passed | 7 passed | 7 passed |
| All application tests | 973 passed | 973 passed | 973 passed |
| Native Desktop and supervised Host smoke | Passed | **Skipped** | Passed |
| Whole job | Passed | Failed: first native reference access | Failed: separate stock CLIP comparison |

Each application total comprises Core 188, Workflow 49, Tokenization 540, Desktop headless 35 and Host 161. The same 33 selected test names occur on all three platforms; every recorded outcome is `Passed`. The selection covers the real prompt compiler, HTTP submission, execution and history routes, including connection precedence, one-level literal envelopes, exact reserved-key spelling, case-distinct node IDs and input names, and preservation of ordinary nested dictionaries. These are product contract regressions, not a new source HTTP oracle or a claim of complete frontend parity. The detailed contract and known boundaries remain in [the local qualification](prompt-literals-host.md).

The successful Windows and macOS smoke steps each emitted:

> ComfySharp Desktop smoke passed: native window, supervised Host, text and CPU sigma graphs, case-sensitive UI history and native preview.

The workflow checks the process exit code as well as this message. At this commit, the smoke uses the actual native window and supervised Host, executes text and CPU sigma graphs, and checks two real PreviewAny outputs with IDs differing only in case through Desktop's history retrieval. Linux never reached that step after its earlier failure; no Linux smoke success is inferred from the headless tests or the other platforms.

Windows additionally passed 574 ordinary Inference tests and the three separately filtered stock CLIP tests. Their test IDs are disjoint from the ordinary test set, giving **1,550 passed tests across seven TRX**. The checkpoint, pipeline and first-use runs overlap ordinary test coverage and are not added. macOS passed its ordinary Inference step but failed its later stock CLIP step; Linux skipped ordinary Inference after the first-use failure. This audit does not reinterpret either native failure, inspect their tensor traces, or attribute a cause to CPU capabilities or library identity.

The [machine-readable evidence](prompt-literals-ci-4faa2ea.json) records all 33 names and outcomes per OS, application project counters, smoke and failure step statuses, 18 independently rehashed TRX, four downloaded artifact identities, metadata/log hashes and 13 source-file pins at the run commit. Eleven files from the earlier local proof—including the three new test files, Host JSON option, both Desktop changes, tested Host project reference and four lock files—match their previously tested canonical bytes exactly. The workflow and prompt compiler are also pinned directly from this commit. GitHub's archive digests are preserved as reported; the extracted TRX hashes were calculated independently.

This was a read-only artifact audit. No test, native calculation, upstream source collection or workflow rerun was performed. The earlier local 245-test result is not added to CI totals. Later TemplateNames changes and source-comparator tests are outside this run.
