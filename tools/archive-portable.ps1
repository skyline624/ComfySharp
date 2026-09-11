param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][ValidateSet('win-x64','linux-x64','osx-arm64')][string]$Runtime
)
$ErrorActionPreference = 'Stop'
$directoryPath = (Resolve-Path -LiteralPath $Directory).Path.TrimEnd('\','/')
$archive = "$directoryPath.tar.gz"
if (Test-Path -LiteralPath $archive) { throw "Archive already exists: $archive. Choose a fresh directory." }
$executables = @('ComfySharp.Desktop', 'host/ComfySharp.Host') | ForEach-Object { if ($Runtime -eq 'win-x64') { "$_.exe" } else { $_ } }
foreach ($executable in $executables) {
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
$memberRoot = Split-Path -Leaf $directoryPath
foreach ($executable in $executables) {
    $expected = "$memberRoot/$executable"
    if (@($members | Where-Object { $_ -ceq $expected }).Count -ne 1) {
        throw "Archive does not contain the expected executable at its exact path: $expected"
    }
}
if ($Runtime -ne 'win-x64') {
    $listing = @(& tar -tvzf $archive)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect portable archive permissions.' }
    foreach ($executable in $executables) {
        $entry = @($listing | Where-Object { $_ -match ('\s' + [regex]::Escape("$memberRoot/$executable") + '$') })
        if ($entry.Count -ne 1 -or $entry[0] -notmatch '^-[rwxstST-]{2}x') {
            throw "Archive does not preserve owner execute permission for $executable. Run publish on its target OS."
        }
    }
}
Write-Output "Local validation archive (redistribution blocked): $archive"
