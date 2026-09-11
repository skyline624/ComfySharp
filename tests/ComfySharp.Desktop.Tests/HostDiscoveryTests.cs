using System.Runtime.InteropServices;
using ComfySharp.Desktop;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public sealed class HostDiscoveryTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "ComfySharp-host-discovery-" + Guid.NewGuid().ToString("N"));
    private static string AppHost => OperatingSystem.IsWindows() ? "ComfySharp.Host.exe" : "ComfySharp.Host";

    [Fact]
    public void ConfiguredPathTakesPrecedenceAndIsNotSilentlyReplacedWhenMissing()
    {
        Touch("host", AppHost);
        var configured = Path.Combine(root, "custom", "unbuilt-host.dll");
        Assert.Equal(configured, HostSupervisor.DiscoverHost(root, configured));
    }

    [Fact]
    public void PublishedHostFolderTakesPrecedenceOverFlatLayout()
    {
        Touch(AppHost);
        var packaged = Touch("host", "ComfySharp.Host.dll");
        Assert.Equal(packaged, HostSupervisor.DiscoverHost(root, ""));
    }

    [Fact]
    public void NativeAppHostTakesPrecedenceOverManagedDllInPublishedFolder()
    {
        Touch("host", "ComfySharp.Host.dll");
        var executable = Touch("host", AppHost);
        Assert.Equal(executable, HostSupervisor.DiscoverHost(root, ""));
    }

    [Fact]
    public void DevelopmentDiscoveryPrefersCurrentCpuBundleToLegacyBuild()
    {
        var runtime = RuntimeInformation.OSArchitecture switch
        {
            Architecture.X64 when OperatingSystem.IsWindows() => "win-x64",
            Architecture.X64 when OperatingSystem.IsLinux() => "linux-x64",
            Architecture.Arm64 when OperatingSystem.IsMacOS() => "osx-arm64",
            _ => throw new PlatformNotSupportedException("The test requires a supported V1 platform.")
        };
        Touch("src", "ComfySharp.Host", "bin", "Release", "net10.0", AppHost);
        var current = Touch("src", "ComfySharp.Host", "bin", "native", runtime, "cpu", "Release", "net10.0", AppHost);
        var desktop = Path.Combine(root, "src", "ComfySharp.Desktop", "bin", "Release", "net10.0");
        Directory.CreateDirectory(desktop);
        Assert.Equal(current, HostSupervisor.DiscoverHost(desktop, ""));
    }

    [Fact]
    public void MissingHostReturnsNull()
    {
        Directory.CreateDirectory(root);
        Assert.Null(HostSupervisor.DiscoverHost(root, ""));
    }

    [Theory]
    [InlineData("Debug", "Release")]
    [InlineData("Release", "Debug")]
    public void EveryCurrentBundlePrecedesLegacyOutputs(string desktopConfiguration, string nativeConfiguration)
    {
        var runtime = SystemRuntime();
        Touch("src", "ComfySharp.Host", "bin", desktopConfiguration, "net10.0", AppHost);
        var current = Touch("src", "ComfySharp.Host", "bin", "native", runtime, "cpu", nativeConfiguration, "net10.0", AppHost);
        var desktop = Path.Combine(root, "src", "ComfySharp.Desktop", "bin", desktopConfiguration, "net10.0");
        Directory.CreateDirectory(desktop);
        Assert.Equal(current, HostSupervisor.DiscoverHost(desktop, ""));
    }

    [Fact]
    public void MatchingDesktopConfigurationPrecedesOtherCurrentConfiguration()
    {
        var runtime = SystemRuntime();
        Touch("src", "ComfySharp.Host", "bin", "native", runtime, "cpu", "Debug", "net10.0", AppHost);
        var release = Touch("src", "ComfySharp.Host", "bin", "native", runtime, "cpu", "Release", "net10.0", AppHost);
        var desktop = Path.Combine(root, "src", "ComfySharp.Desktop", "bin", "Release", "net10.0");
        Directory.CreateDirectory(desktop);
        Assert.Equal(release, HostSupervisor.DiscoverHost(desktop, ""));
    }

    private static string SystemRuntime() => RuntimeInformation.OSArchitecture switch
    {
        Architecture.X64 when OperatingSystem.IsWindows() => "win-x64",
        Architecture.X64 when OperatingSystem.IsLinux() => "linux-x64",
        Architecture.Arm64 when OperatingSystem.IsMacOS() => "osx-arm64",
        _ => throw new PlatformNotSupportedException("The test requires a supported V1 platform.")
    };

    private string Touch(params string[] components)
    {
        var path = Path.Combine([root, .. components]);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "discovery fixture; not executable");
        return path;
    }

    public void Dispose()
    {
        if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
    }
}
