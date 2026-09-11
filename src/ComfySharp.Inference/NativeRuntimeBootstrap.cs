using System.Runtime.InteropServices;

namespace ComfySharp.Inference;

/// <summary>Initializes bundled native dependencies before the first TorchSharp operation.</summary>
public static class NativeRuntimeBootstrap
{
    private static readonly NativeRuntimeInitializer Initializer = new(
        OperatingSystem.IsMacOS(), RuntimeInformation.ProcessArchitecture,
        AppContext.BaseDirectory, File.Exists, NativeLibrary.Load);

    /// <summary>Preloads the macOS ARM64 bundle once. Other platforms require no additional bootstrap.</summary>
    public static void Initialize() => Initializer.Initialize();
}

// The injected filesystem and loader keep platform/path/error policy testable without loading native code.
internal sealed class NativeRuntimeInitializer
{
    private static readonly string[] RequiredLibraries =
        ["libomp.dylib", "libLibTorchSharp.dylib", "libtorch.dylib", "libtorch_cpu.dylib", "libc10.dylib"];
    private readonly Lazy<bool> initialization;
    private readonly List<nint> handles = [];

    internal NativeRuntimeInitializer(bool isMacOS, Architecture architecture, string baseDirectory,
        Func<string, bool> fileExists, Func<string, nint> load)
    {
        initialization = new(() =>
        {
            if (!isMacOS || architecture != Architecture.Arm64) return true;
            if (!Path.IsPathFullyQualified(baseDirectory))
                throw new ArgumentException("The native application directory must be an absolute path.", nameof(baseDirectory));

            string applicationDirectory = Path.GetFullPath(baseDirectory);
            string[] candidates = [Path.Combine(applicationDirectory, "runtimes", "osx-arm64", "native"), applicationDirectory];
            var missingBundles = new List<string>();
            string? nativeDirectory = null;
            foreach (string candidate in candidates)
            {
                var missing = RequiredLibraries.Where(name => !fileExists(Path.Combine(candidate, name))).ToArray();
                if (missing.Length == 0)
                {
                    nativeDirectory = candidate;
                    break;
                }
                missingBundles.Add($"'{candidate}': missing {string.Join(", ", missing)}");
            }
            if (nativeDirectory is null)
                throw new DllNotFoundException("Bundled macOS ARM64 native runtime is missing or incomplete. " +
                    "All native libraries must be siblings in the build or publish output. Checked " + string.Join("; ", missingBundles));

            // libtorch_cpu imports the Homebrew install name also recorded by the bundled libomp.
            // Loading that exact bundled library first lets dyld satisfy the dependency without an installation.
            foreach (string name in new[] { "libomp.dylib", "libLibTorchSharp.dylib" })
            {
                string path = Path.Combine(nativeDirectory, name);
                try { handles.Add(load(path)); }
                catch (Exception error) when (error is DllNotFoundException or BadImageFormatException or FileLoadException)
                {
                    throw new DllNotFoundException($"Failed to initialize bundled macOS ARM64 native runtime while loading '{path}': {error.Message}", error);
                }
            }
            return true;
        }, LazyThreadSafetyMode.ExecutionAndPublication);
    }

    // Keep successful handles, including an OpenMP handle before a bridge failure, for process lifetime.
    // TorchSharp can cache native entry points; unloading here would invalidate them. Lazy also caches failures.
    internal void Initialize() => _ = initialization.Value;
}
