param(
    [ValidateSet('win-x64','linux-x64','osx-arm64')][string]$Runtime = 'win-x64',
    [string]$OutputDirectory = 'artifacts/portable'
)
$ErrorActionPreference = 'Stop'
Write-Warning 'LOCAL VALIDATION ONLY: dependency notices and an exact distribution SBOM are incomplete. Do not redistribute these binaries. See docs/packaging.md.'
$repoRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = if ([IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $repoRoot $OutputDirectory }
$destination = [IO.Path]::GetFullPath((Join-Path $outputRoot $Runtime))
if (Test-Path -LiteralPath $destination) { throw "Output already exists: $destination. Choose a fresh directory." }
$projects = @('ComfySharp.Host','ComfySharp.Desktop')
foreach ($project in $projects) {
    & dotnet publish (Join-Path $repoRoot "src/$project/$project.csproj") -c Release -r $Runtime --self-contained true -p:RestoreLockedMode=true -p:PublishSingleFile=false --artifacts-path (Join-Path $outputRoot '.build') -o $destination
    if ($LASTEXITCODE -ne 0) { throw "Publish failed for $project." }
}
Copy-Item -LiteralPath (Join-Path $repoRoot 'LICENSE') -Destination $destination
Copy-Item -LiteralPath (Join-Path $repoRoot 'THIRD_PARTY_NOTICES.md') -Destination $destination
Copy-Item -LiteralPath (Join-Path $repoRoot 'docs/STATUS.md') -Destination $destination
$packageDocs = Join-Path $destination 'docs'
New-Item -ItemType Directory -Path $packageDocs -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $repoRoot 'docs/packaging.md') -Destination $packageDocs
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
Write-Output 'This package is an initial editor/Host build; no image/video/audio/3D model is supported yet.'
