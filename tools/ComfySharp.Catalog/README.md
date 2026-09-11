# Catalogue qualification

`dotnet run --project tools/ComfySharp.Catalog -- docs/capabilities/manifest.json --release`
requires `catalogueComplete: true` and every local capability to be implemented,
unit tested, and exercised by a real workflow. Every `platforms` object must contain
boolean `true` for `win-x64-cpu`, `win-x64-cuda`, `linux-x64-cpu`, `linux-x64-cuda`,
`osx-arm64-cpu`, and `osx-arm64-mps`.

Each platform must also have an object in `evidence` with its exact `platform` key,
nonblank `hardware`, `scenario`, and `artifact` strings, and boolean `passed: true`.
The artifact identifies the durable run report. Empty objects and arbitrary JSON
values do not qualify. This validates evidence structure, not its authenticity or
hardware parity; the release audit must examine the referenced reports.

`dotnet run --project tools/ComfySharp.Catalog -- docs/capabilities/manifest.json --node-evidence docs/capabilities/node-registration.json docs/capabilities/node-schemas.json`
cross-checks the registration and parameter evidence against the manifest: pinned
sources, module order, candidate counts, retained unregistered declarations,
schema identities and unresolved expressions. This detects drift between the
documents; source enumeration and runtime schema finalization still need their
own evidence. A successful check does not set `catalogueComplete` or qualify a model.
