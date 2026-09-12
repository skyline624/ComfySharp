param(
    [Parameter(Mandatory)][string]$Archive,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [Parameter(Mandatory)][string]$NativeBundleRecipe,
    [ValidateSet('nuget','composed')][string]$NativeSource = 'nuget'
)
$ErrorActionPreference = 'Stop'
if (-not $IsLinux) { throw 'This loaded-module qualification requires Linux /proc.' }
$archivePath = (Resolve-Path -LiteralPath $Archive).Path
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $output) { throw 'Qualification requires a new output directory.' }
$recipe = Get-Content -LiteralPath $NativeBundleRecipe -Raw | ConvertFrom-Json
if ($recipe.runtime -cne 'linux-x64') { throw 'Expected a linux-x64 native recipe.' }
New-Item -ItemType Directory -Path $output | Out-Null
$extracted = Join-Path $output 'extracted'
New-Item -ItemType Directory -Path $extracted | Out-Null
& tar -xzf $archivePath -C $extracted
if ($LASTEXITCODE -ne 0) { throw 'Archive extraction failed.' }
$package = Join-Path $extracted 'linux-x64-cpu'
$engineDirectory = Join-Path $package 'host'
Push-Location $package
try {
    & sha256sum --check --quiet SHA256SUMS
    if ($LASTEXITCODE -ne 0) { throw 'Extracted package checksums differ.' }
} finally { Pop-Location }
$emptyPath = Join-Path $output 'empty-path'
New-Item -ItemType Directory -Path $emptyPath | Out-Null
$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0)
$listener.Start(); $port = $listener.LocalEndpoint.Port; $listener.Stop()
$uri = "http://127.0.0.1:$port"
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = Join-Path $engineDirectory 'ComfySharp.Host'
$start.WorkingDirectory = $package
$start.UseShellExecute = $false
$start.RedirectStandardOutput = $true; $start.RedirectStandardError = $true
foreach ($arg in @('--urls',$uri,'--data-dir',(Join-Path $output 'host-data'),'--cpu-threads','1')) { $start.ArgumentList.Add($arg) }
foreach ($key in @('PATH','DOTNET_ROOT','DOTNET_ROOT_X64')) { $start.Environment[$key] = $emptyPath }
$start.Environment['DOTNET_MULTILEVEL_LOOKUP'] = '0'
$child = [Diagnostics.Process]::new(); $child.StartInfo = $start
$started = $false; $loaded = @(); $preview = $null
try {
    $started = $child.Start()
    if (-not $started) { throw 'Extracted Host failed to start.' }
    $stdout = $child.StandardOutput.ReadToEndAsync(); $stderr = $child.StandardError.ReadToEndAsync()
    $deadline = [DateTime]::UtcNow.AddSeconds(60); $ready = $false
    while ([DateTime]::UtcNow -lt $deadline -and -not $child.HasExited) {
        try { $health = Invoke-RestMethod "$uri/health" -TimeoutSec 2; $ready = $health.product -ceq 'ComfySharp'; if ($ready) { break } } catch { }
        Start-Sleep -Milliseconds 100
    }
    if (-not $ready) { throw 'Extracted Host did not become ready with an empty executable search path.' }
    $prompt = Get-Content -LiteralPath (Join-Path (Split-Path -Parent $PSScriptRoot) 'examples/sigma-preview.prompt.json') -Raw
    $accepted = Invoke-RestMethod "$uri/prompt" -Method Post -ContentType application/json -Body $prompt -TimeoutSec 10
    if (-not $accepted.prompt_id) { throw 'Host did not accept the sigma workflow.' }
    $entry = $null
    while ([DateTime]::UtcNow -lt $deadline -and -not $child.HasExited) {
        $history = Invoke-RestMethod "$uri/history" -TimeoutSec 2
        $entry = $history.PSObject.Properties[$accepted.prompt_id].Value
        if ($null -ne $entry) { break }
        Start-Sleep -Milliseconds 100
    }
    if ($entry.status.completed -ne $true -or $entry.outputs.'3'.text[0] -cne 'tensor([3., 2.])' -or $entry.outputs.'4'.text[0] -cne 'tensor([2., 1., 0.])') {
        throw 'Extracted Host sigma workflow did not reproduce both expected tensor previews.'
    }
    $preview = $entry.outputs
    $paths = @(Get-Content -LiteralPath "/proc/$($child.Id)/maps" | ForEach-Object {
        $parts = $_ -split '\s+',6
        if ($parts.Count -eq 6 -and $parts[5].StartsWith('/')) { $parts[5] }
    } | Sort-Object -Unique)
    if ($paths | Where-Object { [IO.Path]::GetFileName($_) -match '^(libpython|libnode)' }) { throw 'An interpreter was loaded in the extracted Host.' }
    $images = @($recipe.files | Where-Object role -CEQ 'native')
    foreach ($file in $images) {
        $name = $file.destination.Substring(7)
        $names = @($name) + @($file.aliases)
        $found = @($paths | Where-Object { [IO.Path]::GetFileName($_) -cin $names })
        if ($found.Count -ne 1) { throw "Expected exactly one loaded native image for $name, found $($found.Count)." }
        if ([IO.Path]::GetDirectoryName($found[0]) -cne $engineDirectory) { throw 'Native code loaded outside the extracted Host closure.' }
        $hash = (Get-FileHash -LiteralPath $found[0] -Algorithm SHA256).Hash.ToLowerInvariant()
        $expected = if ($NativeSource -eq 'composed') { @($file.sha256) } else { @($file.replacesSha256) }
        if ($hash -cnotin $expected) { throw "Unexpected loaded hash for $name." }
        $loaded += @{ name = [IO.Path]::GetFileName($found[0]); sha256 = $hash }
    }
    foreach ($name in @($recipe.binding.name,'libcoreclr.so','libhostpolicy.so')) {
        $found = @($paths | Where-Object { [IO.Path]::GetFileName($_) -ceq $name })
        if ($found.Count -ne 1 -or [IO.Path]::GetDirectoryName($found[0]) -cne $engineDirectory) { throw "Missing or external self-contained dependency: $name." }
        $hash = (Get-FileHash -LiteralPath $found[0] -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($name -ceq $recipe.binding.name -and $hash -cne $recipe.binding.sha256) { throw 'TorchSharp binding differs.' }
        $loaded += @{ name = $name; sha256 = $hash }
    }
} finally {
    if ($started) {
        if (-not $child.HasExited) { $child.Kill($true) }
        if (-not $child.WaitForExit(10000)) { throw 'Owned Host process failed to terminate.' }
        $stdout.GetAwaiter().GetResult() | Set-Content -LiteralPath (Join-Path $output 'host.stdout.log')
        $stderr.GetAwaiter().GetResult() | Set-Content -LiteralPath (Join-Path $output 'host.stderr.log')
    }
    $child.Dispose()
}

# Xvfb itself needs shell utilities. Only the application and its supervised Host
# receive an empty executable search path and absent SDK root.
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = '/usr/bin/xvfb-run'; $start.WorkingDirectory = $package; $start.UseShellExecute = $false
$start.RedirectStandardOutput = $true; $start.RedirectStandardError = $true
foreach ($arg in @('-a','-s','-screen 0 1280x800x24','/usr/bin/env',"PATH=$emptyPath","DOTNET_ROOT=$emptyPath","DOTNET_ROOT_X64=$emptyPath",'DOTNET_MULTILEVEL_LOOKUP=0','COMFYSHARP_HOST_PATH=',(Join-Path $package 'ComfySharp.Desktop'),'--smoke-test')) { $start.ArgumentList.Add($arg) }
$child = [Diagnostics.Process]::new(); $child.StartInfo = $start
$started = $false; $limit = [Threading.CancellationTokenSource]::new([TimeSpan]::FromMinutes(2))
try {
    $started = $child.Start(); if (-not $started) { throw 'Extracted Desktop did not start.' }
    $stdout = $child.StandardOutput.ReadToEndAsync(); $stderr = $child.StandardError.ReadToEndAsync()
    [void]$child.WaitForExitAsync($limit.Token).GetAwaiter().GetResult()
    $text = $stdout.GetAwaiter().GetResult(); $errors = $stderr.GetAwaiter().GetResult()
    $text | Set-Content -LiteralPath (Join-Path $output 'desktop.stdout.log'); $errors | Set-Content -LiteralPath (Join-Path $output 'desktop.stderr.log')
    if ($child.ExitCode -ne 0 -or $text -notmatch 'ComfySharp Desktop smoke passed:') { throw "Extracted Desktop/Host smoke failed (exit $($child.ExitCode))." }
} finally {
    if ($started -and -not $child.HasExited) { $child.Kill($true); [void]$child.WaitForExit(10000) }
    $limit.Dispose(); $child.Dispose()
}
$receiptPath = Join-Path $engineDirectory 'comfysharp-native-bundle.json'
if ((Test-Path -LiteralPath $receiptPath) -ne ($NativeSource -eq 'composed')) { throw 'Unexpected composition receipt presence.' }
$receipt = if ($NativeSource -eq 'composed') { Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json } else { $null }
@{ success = $true; nativeSource = $NativeSource; archiveSha256 = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant();
    loadedImages = $loaded; previews = $preview; composition = $receipt; emptyApplicationPath = $true; selfContainedRuntimeLoaded = $true;
    desktopWithSupervisedHostPassed = $true; scope = 'Portable Linux CPU execution and native origins; no model-family, GPU or distribution release qualification.'
} | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $output 'result.json')
Write-Output "Extracted portable qualification passed: $NativeSource, native Host sigma results, exact loaded images, supervised Desktop, empty application PATH and no SDK root."
