# ComfySharp engineering contract

- This directory is an independent repository. Target all Git operations here. Never change the parent ComfyUI sources or its Git history/remotes.
- Implement the approved scope in docs/MIGRATION.md. Preserve upstream class_type identifiers and data contracts. Record incomplete capabilities honestly; synthetic tests are not evidence of model-family parity.
- Product code and distributed tests use C#/.NET and native libraries only. No Python interpreter, pip, Node runtime, JavaScript frontend, model downloads, telemetry or outbound Internet calls in the product.
- Desktop and Host are separate processes. Desktop owns documents; Host owns execution and native resources. Do not ship fake successful inference.
- Keep documents and compilers independent of Avalonia. Retain unknown fields and unavailable third-party nodes with explicit diagnostics.
- Dispose native resources deterministically. Cancellation must target a specific job; never cancel its successor.
- Keep weights, personal data, credentials, build outputs and local audit paths out of Git. Use pinned public upstream references for provenance.
- Use central package versions and checked-in package lock files. Add meaningful behavior tests for contract, scheduling, parsing and lifetime changes.
- Use codex/<lot>-<feature> work branches. Do not claim V1 readiness until every mandatory capability has hardware-backed evidence on all required platforms.
- On this Windows workstation prefix shell commands with rtk (rtk proxy for unrecognized commands).
