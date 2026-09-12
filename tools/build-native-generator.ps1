param(
    [Parameter(Mandatory)][string]$TorchSdkRoot,
    [Parameter(Mandatory)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
if (-not $IsWindows) { throw 'This build helper requires Windows x64 and Visual Studio C++ Build Tools. Use the CMake project for other toolchains.' }
$sdk = (Resolve-Path -LiteralPath $TorchSdkRoot).Path
foreach ($relative in @('include/ATen/Context.h','include/ATen/Version.h','include/torch/csrc/api/include/torch/version.h','lib/torch_cpu.lib','lib/c10.lib')) {
    if (-not (Test-Path -LiteralPath (Join-Path $sdk $relative))) { throw "Missing libtorch SDK input: $relative" }
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -ne 0 -or -not $vsRoot) { throw 'Visual Studio x64 C++ Build Tools were not found.' }
Import-Module (Join-Path $vsRoot 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vsRoot -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
$destination = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$source = Join-Path $PSScriptRoot '../native/ComfySharp.NativeGenerator/generator.cpp'
& cl.exe /nologo /std:c++17 /EHsc /MD /O2 /LD /I"$sdk/include" /I"$sdk/include/torch/csrc/api/include" $source /Fo"$destination/generator.obj" /link /LIBPATH:"$sdk/lib" torch_cpu.lib c10.lib /OUT:"$destination/ComfySharp.Native.dll" /IMPLIB:"$destination/ComfySharp.Native.lib"
if ($LASTEXITCODE -ne 0) { throw 'Native generator compilation failed.' }
Get-Item -LiteralPath (Join-Path $destination 'ComfySharp.Native.dll')
