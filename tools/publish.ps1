param(
    [ValidateSet('win-x64','linux-x64','osx-arm64')][string]$Runtime = 'win-x64',
    [ValidateSet('cpu','cuda')][string]$NativeBackend = 'cpu',
    [string]$OutputDirectory = 'artifacts/portable',
    [string]$NativeBundleDirectory,
    [string]$NativeBundleRecipe
)
$ErrorActionPreference = 'Stop'
if ($NativeBackend -eq 'cuda' -and $Runtime -ne 'win-x64') { throw 'The configured CUDA 12.8 bundle currently targets win-x64 only. Linux CUDA qualification remains pending.' }
$composeNative = -not [string]::IsNullOrWhiteSpace($NativeBundleDirectory)
if ($composeNative -ne (-not [string]::IsNullOrWhiteSpace($NativeBundleRecipe))) { throw 'NativeBundleDirectory and NativeBundleRecipe must be supplied together.' }
if ($composeNative) {
    if ($Runtime -ne 'linux-x64' -or $NativeBackend -ne 'cpu') { throw 'Explicit native bundle packaging currently supports linux-x64 CPU only.' }
    $NativeBundleDirectory = (Resolve-Path -LiteralPath $NativeBundleDirectory).Path
    $NativeBundleRecipe = (Resolve-Path -LiteralPath $NativeBundleRecipe).Path
    $recipe = Get-Content -LiteralPath $NativeBundleRecipe -Raw | ConvertFrom-Json
    if ($recipe.runtime -cne $Runtime) { throw 'Native bundle recipe targets a different runtime.' }
}
Write-Warning 'LOCAL VALIDATION ONLY: dependency notices and an exact distribution SBOM are incomplete. Do not redistribute these binaries. See docs/packaging.md.'
$repoRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = if ([IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $repoRoot $OutputDirectory }
$destination = [IO.Path]::GetFullPath((Join-Path $outputRoot "$Runtime-$NativeBackend"))
if (Test-Path -LiteralPath $destination) { throw "Output already exists: $destination. Choose a fresh directory." }
if (Test-Path -LiteralPath "$destination.tar.gz") { throw 'Output archive already exists. Choose a fresh directory.' }
$buildRoot = [IO.Path]::GetFullPath((Join-Path $outputRoot ".build/$Runtime-$NativeBackend"))
if ($composeNative) {
    $comparison = if ($IsWindows) { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
    $bundleRoot = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath($NativeBundleDirectory))
    foreach ($target in @($destination,$buildRoot)) {
        if ($target.Equals($bundleRoot,$comparison) -or $target.StartsWith($bundleRoot + [IO.Path]::DirectorySeparatorChar,$comparison) -or $bundleRoot.StartsWith($target + [IO.Path]::DirectorySeparatorChar,$comparison)) {
            throw 'Prepared bundle and publish/build directories must not overlap.'
        }
    }
    $originalHost = Join-Path $buildRoot 'original-host'
    if (Test-Path -LiteralPath $originalHost) { throw 'Original Host staging already exists. Choose a fresh output root.' }
    & dotnet build (Join-Path $repoRoot 'tools/ComfySharp.NativeBundle/ComfySharp.NativeBundle.csproj') -c Release -p:RestoreLockedMode=true
    if ($LASTEXITCODE -ne 0) { throw 'Native bundle tool build failed.' }
    $bundleTool = Join-Path $repoRoot 'tools/ComfySharp.NativeBundle/bin/Release/net10.0/ComfySharp.NativeBundle.dll'
}
$projects = @('ComfySharp.Host','ComfySharp.Desktop')
foreach ($project in $projects) {
    $projectDestination = if ($project -eq 'ComfySharp.Host') { Join-Path $destination 'host' } else { $destination }
    $publishDestination = if ($composeNative -and $project -eq 'ComfySharp.Host') { $originalHost } else { $projectDestination }
    & dotnet publish (Join-Path $repoRoot "src/$project/$project.csproj") -c Release -r $Runtime --self-contained true -p:NativeRuntime=$Runtime -p:NativeBackend=$NativeBackend -p:RestoreLockedMode=true -p:PublishSingleFile=false --artifacts-path $buildRoot -o $publishDestination
    if ($LASTEXITCODE -ne 0) { throw "Publish failed for $project." }
    if ($composeNative -and $project -eq 'ComfySharp.Host') {
        & dotnet $bundleTool compose --recipe $NativeBundleRecipe --bundle $NativeBundleDirectory --application $originalHost --output $projectDestination
        if ($LASTEXITCODE -ne 0) { throw 'Native Host composition failed; no portable archive was created.' }
    }
}
if (Get-ChildItem -LiteralPath $destination -File | Where-Object { $_.Name -match '^(lib)?torch_cpu\.(dll|so|dylib)$' }) {
    throw 'A native libtorch payload escaped the separate host dependency folder.'
}
Copy-Item -LiteralPath (Join-Path $repoRoot 'LICENSE') -Destination $destination
Copy-Item -LiteralPath (Join-Path $repoRoot 'THIRD_PARTY_NOTICES.md') -Destination $destination
Copy-Item -LiteralPath (Join-Path $repoRoot 'docs/STATUS.md') -Destination $destination
$packageDocs = Join-Path $destination 'docs'
New-Item -ItemType Directory -Path $packageDocs -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $repoRoot 'docs/packaging.md') -Destination $packageDocs
if ($composeNative) { Copy-Item -LiteralPath (Join-Path $repoRoot 'docs/NATIVE_BUNDLES.md') -Destination $packageDocs }
$files = @(Get-ChildItem -LiteralPath $destination -File -Recurse | Sort-Object FullName)
$checksums = $files | ForEach-Object {
    $stream = [IO.File]::OpenRead($_.FullName)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { $hash = [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
    $relative = $_.FullName.Substring($destination.TrimEnd('\','/').Length + 1).Replace('\','/')
    "$hash  $relative"
}
[IO.File]::WriteAllLines((Join-Path $destination 'SHA256SUMS'), [string[]]$checksums, [Text.UTF8Encoding]::new($false))
& (Join-Path $PSScriptRoot 'archive-portable.ps1') -Directory $destination -Runtime $Runtime
Write-Output "Local unsigned validation build: $destination"
Write-Output "Native distribution bundle: $Runtime / $NativeBackend. Bundle selection is not model or GPU qualification."
if ($composeNative) { Write-Output "Explicit native composition: $($recipe.id). Receipt and notices are inside host/." }
Write-Output 'This package contains the current editor/Host and SD1.5 development workflow; no complete model family is qualified.'
