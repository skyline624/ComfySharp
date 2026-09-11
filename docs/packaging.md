# Local portable packaging validation

`tools/publish.ps1 -Runtime win-x64` (or `linux-x64`, `osx-arm64` on
the corresponding OS) builds the self-contained Desktop and Host with locked
restore. Use a fresh `-OutputDirectory` on each run. The script writes per-file
SHA-256 checksums and a sibling `.tar.gz` archive. Archive validation rejects
missing apphosts and, on Unix targets, missing owner execute permission for
Desktop or Host. The tar archive preserves Unix modes; uploading the raw publish
directory through GitHub artifact storage does not.

**These outputs are for local validation only. Binary redistribution is blocked.**
The manually dispatched portable workflow builds and validates archives on the
three target operating systems but deliberately has no binary upload step. The
normal CI workflow separately builds and tests the product. Source publication
does not depend on completion of binary packaging.

## Unresolved binary distribution requirements

The copied project GPL and `THIRD_PARTY_NOTICES.md` are not a complete set of
dependency license texts. No distribution SBOM is generated yet. The blocking
work is to establish and verify the exact published dependency closure for both
applications on each RID, including self-contained .NET/ASP.NET runtime packs
and native assets, and then ship all applicable license and notice texts.

NuGet license metadata alone is insufficient: for example, Avalonia 12.0.5's
cached package declares a license without including a standalone license text;
SkiaSharp.NativeAssets.Linux 3.119.4 includes both `LICENSE.txt` and a substantial
`THIRD-PARTY-NOTICES.txt`. Both applicable texts, including notices for embedded
native components, must be retained. Resolve license-expression or URL-only
packages to authoritative texts for the exact version, recording provenance.
Audit the actual published SQLite, Skia, Avalonia native assets and runtime packs,
rather than assuming that the top-level managed package license covers them.

Before enabling any binary artifact upload or release:

- Collect exact versions and package hashes from the RID-specific resolved
  assets/locks, including runtime packs selected by the SDK. Map the closure to
  published files and retain the applicable dependency license and notice texts.
- Produce a schema-valid SPDX or CycloneDX SBOM with exact component versions,
  package identities, cryptographic hashes, dependency relationships and license
  provenance. Include the distribution's file hashes. `SHA256SUMS` alone is not
  an SBOM, and the general dependency list is not evidence of notice completeness.
- Verify notice coverage and SBOM content for every RID, test the downloaded
  archive's contents and Desktop/Host executable modes on Linux and macOS, and
  explicitly remove the local-only restriction as part of that reviewed change.
- Upload only the validated `.tar.gz` files, never their raw source directories.

TorchSharp/libtorch probe packages are separate from the current Desktop/Host
publish. Any future distribution that includes them needs its own applicable
native notices and SBOM coverage. Do not infer binary inclusion from the solution
dependency list alone.
