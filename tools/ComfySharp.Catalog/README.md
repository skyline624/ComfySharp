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

The node-evidence check also recounts unresolved values at their canonical input,
output and metadata locations, checks every unresolved-field mirror and per-node
summary, and verifies parameter resolution statuses. Mirrored summaries are not
counted a second time.

When schemas contain static resolution evidence, the command requires the sibling
`node-schema-resolutions.json`. Its reviewed JSON identity is fixed in
`NodeSchemaStaticResolution.PlanSha256` (compact UTF-8 JSON, property order retained,
no ASCII escaping; whitespace and LF/CRLF do not affect identity). The plan covers
exactly 55 occurrences and 17 expressions at backend
`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`: IO enum members, ordered scheduler keys,
and PorterDuff enum names. The checker verifies the applied values, original
expressions, source locations, shared IO enum declarations and all 55 evidence
records. Removing the plan or changing a reviewed literal is an error.

`NodeSchemaStaticResolution.Apply` is a pure JSON transformation for this reviewed
plan. It clones the input, requires each original symbolic value and its mirror,
and publishes only a complete result. Already-applied input is rejected. It does
not import Python, evaluate expressions, discover files/devices, or execute nodes.
Source blob hashes in the plan attest the reviewed Git declarations; the command
does not fetch or rehash absent upstream blobs. Its success establishes document
consistency with that plan, not authenticity of a runtime schema, node execution,
or numerical/model qualification. The remaining 226 occurrences stay explicit;
all capability implementation and qualification flags are unchanged.
