using System.Runtime.InteropServices;
using ComfySharp.Inference;
using Xunit;

namespace ComfySharp.Inference.Tests;

public sealed class NativeRuntimeBootstrapTests
{
    private static readonly string ApplicationDirectory = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "comfysharp-bootstrap-test", "application"));
    private static readonly string NativeDirectory = Path.Combine(ApplicationDirectory, "runtimes", "osx-arm64", "native");
    private static readonly string[] Libraries =
        ["libomp.dylib", "libLibTorchSharp.dylib", "libtorch.dylib", "libtorch_cpu.dylib", "libc10.dylib"];

    [Theory]
    [InlineData(false, Architecture.Arm64)]
    [InlineData(false, Architecture.X64)]
    [InlineData(true, Architecture.X64)]
    public void OtherPlatformsDoNotInspectOrLoadNativeFiles(bool isMacOS, Architecture architecture)
    {
        var initializer = new NativeRuntimeInitializer(isMacOS, architecture, ApplicationDirectory,
            _ => throw new Exception("Unexpected filesystem access."), _ => throw new Exception("Unexpected native load."));
        initializer.Initialize();
        initializer.Initialize();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompleteBuildAndPublishBundlesLoadOpenMpBeforeBridge(bool flatPublish)
    {
        string directory = flatPublish ? ApplicationDirectory : NativeDirectory;
        var bundle = Libraries.Select(name => Path.Combine(directory, name)).ToHashSet(StringComparer.Ordinal);
        var inspected = new List<string>();
        var loaded = new List<string>();
        var initializer = new NativeRuntimeInitializer(true, Architecture.Arm64, ApplicationDirectory,
            path => { inspected.Add(path); return bundle.Contains(path); },
            path => { loaded.Add(path); return (nint)loaded.Count; });

        initializer.Initialize();
        initializer.Initialize();

        Assert.Equal(new[] { Path.Combine(directory, "libomp.dylib"), Path.Combine(directory, "libLibTorchSharp.dylib") }, loaded);
        Assert.Equal(flatPublish ? 10 : 5, inspected.Count);
        Assert.All(inspected, path =>
        {
            Assert.True(Path.IsPathFullyQualified(path));
            Assert.Contains(Path.GetDirectoryName(path), new[] { NativeDirectory, ApplicationDirectory });
        });
    }

    [Fact]
    public void CompleteBuildBundleTakesPriorityOverFlatPublishBundle()
    {
        var loaded = new List<string>();
        var initializer = new NativeRuntimeInitializer(true, Architecture.Arm64, ApplicationDirectory,
            _ => true, path => { loaded.Add(path); return (nint)loaded.Count; });

        initializer.Initialize();

        Assert.All(loaded, path => Assert.Equal(NativeDirectory, Path.GetDirectoryName(path)));
        Assert.Equal(2, loaded.Count);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MissingOrSplitBundlesFailBeforeLoadingAndCacheTheDiagnostic(bool splitBundle)
    {
        int inspections = 0;
        var initializer = new NativeRuntimeInitializer(true, Architecture.Arm64, ApplicationDirectory,
            path =>
            {
                inspections++;
                // The combined files are complete, but neither directory contains a complete bundle.
                return splitBundle && (Path.GetDirectoryName(path) == NativeDirectory
                    ? Path.GetFileName(path) != "libLibTorchSharp.dylib"
                    : Path.GetFileName(path) == "libLibTorchSharp.dylib");
            }, _ => throw new Exception("Incomplete bundles must not be loaded."));

        var failure = Assert.Throws<DllNotFoundException>(initializer.Initialize);
        Assert.Same(failure, Assert.Throws<DllNotFoundException>(initializer.Initialize));
        Assert.Contains("macOS ARM64", failure.Message);
        Assert.Contains(NativeDirectory, failure.Message);
        Assert.Contains(ApplicationDirectory, failure.Message);
        Assert.Contains("libLibTorchSharp.dylib", failure.Message);
        Assert.Contains("libomp.dylib", failure.Message);
        Assert.Equal(10, inspections);
    }

    [Theory]
    [InlineData("libomp.dylib", 1)]
    [InlineData("libLibTorchSharp.dylib", 2)]
    public void LoaderFailurePreservesDyldDiagnosticWithoutRetryOrAlternateBundle(string failingLibrary, int expectedLoads)
    {
        var dyldError = new DllNotFoundException("dyld: Library not loaded: /opt/homebrew/opt/libomp/lib/libomp.dylib");
        var loaded = new List<string>();
        var initializer = new NativeRuntimeInitializer(true, Architecture.Arm64, ApplicationDirectory,
            _ => true, path =>
            {
                loaded.Add(path);
                if (Path.GetFileName(path) == failingLibrary) throw dyldError;
                return (nint)loaded.Count;
            });

        var failure = Assert.Throws<DllNotFoundException>(initializer.Initialize);
        Assert.Same(failure, Assert.Throws<DllNotFoundException>(initializer.Initialize));
        Assert.Same(dyldError, failure.InnerException);
        Assert.Contains(Path.Combine(NativeDirectory, failingLibrary), failure.Message);
        Assert.Contains(dyldError.Message, failure.Message);
        Assert.Equal(expectedLoads, loaded.Count);
        Assert.All(loaded, path => Assert.Equal(NativeDirectory, Path.GetDirectoryName(path)));
    }

    [Fact]
    public void RelativeApplicationDirectoryCannotUseTheWorkingDirectory()
    {
        var initializer = new NativeRuntimeInitializer(true, Architecture.Arm64, "relative-application",
            _ => throw new Exception("Unexpected filesystem access."), _ => throw new Exception("Unexpected native load."));

        Assert.Throws<ArgumentException>(initializer.Initialize);
    }

    [Fact]
    public async Task ConcurrentInitializationLoadsTheBundleOnce()
    {
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        int inspections = 0;
        var loaded = new List<string>();
        var initializer = new NativeRuntimeInitializer(true, Architecture.Arm64, ApplicationDirectory,
            _ => { Interlocked.Increment(ref inspections); return true; }, path =>
            {
                loaded.Add(path);
                entered.Set();
                if (!release.Wait(TimeSpan.FromSeconds(10))) throw new TimeoutException("Test did not release native load.");
                return (nint)loaded.Count;
            });
        var callers = Enumerable.Range(0, 16).Select(_ => Task.Run(initializer.Initialize)).ToArray();
        try { Assert.True(entered.Wait(TimeSpan.FromSeconds(10)), "No caller entered native loading."); }
        finally { release.Set(); }
        await Task.WhenAll(callers);
        initializer.Initialize();

        Assert.Equal(5, inspections);
        Assert.Equal(new[] { Path.Combine(NativeDirectory, "libomp.dylib"), Path.Combine(NativeDirectory, "libLibTorchSharp.dylib") }, loaded);
    }
}
