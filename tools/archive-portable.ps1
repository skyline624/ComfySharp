param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][ValidateSet('win-x64','linux-x64','osx-arm64')][string]$Runtime
)
$ErrorActionPreference = 'Stop'
$directoryPath = (Resolve-Path -LiteralPath $Directory).Path.TrimEnd('\','/')
$archive = "$directoryPath.tar.gz"
if (Test-Path -LiteralPath $archive) { throw "Archive already exists: $archive. Choose a fresh directory." }
foreach ($name in @('ComfySharp.Desktop','ComfySharp.Host')) {
    $executable = if ($Runtime -eq 'win-x64') { "$name.exe" } else { $name }
    if (-not (Test-Path -LiteralPath (Join-Path $directoryPath $executable) -PathType Leaf)) {
        throw "Missing portable executable: $executable"
    }
}
# tar stores Unix modes, unlike uploading a raw directory with upload-artifact.
# Use a relative member root so extraction never writes to a build-machine path.
& tar -czf $archive -C (Split-Path -Parent $directoryPath) (Split-Path -Leaf $directoryPath)
if ($LASTEXITCODE -ne 0) { throw 'Portable archive creation failed.' }
$members = @(& tar -tzf $archive)
if ($LASTEXITCODE -ne 0 -or $members.Count -eq 0) { throw 'Portable archive is empty or unreadable.' }
if ($Runtime -ne 'win-x64') {
    $listing = @(& tar -tvzf $archive)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect portable archive permissions.' }
    foreach ($name in @('ComfySharp.Desktop','ComfySharp.Host')) {
        $entry = @($listing | Where-Object { $_ -match ('/' + [regex]::Escape($name) + '$') })
        if ($entry.Count -ne 1 -or $entry[0] -notmatch '^-[rwxstST-]{2}x') {
            throw "Archive does not preserve owner execute permission for $name. Run publish on its target OS."
        }
    }
}
Write-Output "Local validation archive (redistribution blocked): $archive"
