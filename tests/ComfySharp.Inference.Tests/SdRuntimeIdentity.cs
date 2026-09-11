using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Observational evidence only. Call after a completed forward, never from an operator callback.</summary>
internal static class SdRuntimeIdentity
{
    internal delegate (int Eax, int Ebx, int Ecx, int Edx) CpuIdQuery(int leaf, int subleaf);
    internal sealed record CpuEvidence(string status, string? vendor, string? brand,
        IReadOnlyDictionary<string, bool?> hardwareFeatures, string? errorType);
    internal sealed record RequestedCapability(string status, string? value);
    internal sealed record LibraryIdentity(int instance, string name, string status, long? bytes, string? sha256, string? errorType);
    internal sealed record LibraryEvidence(string status, string capturePoint, IReadOnlyList<LibraryIdentity> libraries, string? errorType);
    private sealed record FileEvidence(string status, long? bytes, string? sha256, string? errorType);

    // Creating the Lazy does no enumeration/hashing. The first traced forward has
    // completed before CaptureAfterForward evaluates it; later cases reuse that snapshot.
    private static readonly Lazy<LibraryEvidence> Libraries = LibraryCache(
        () => ObserveLibraries(LoadedModulePaths, path => File.OpenRead(path)));
    private static readonly Lazy<CpuEvidence> Cpu = new(() => ObserveCpu(X86Base.IsSupported, X86Base.CpuId));

    internal static object CaptureAfterForward() => new
    {
        diagnosticOnly = true,
        capturePoint = "after_first_forward_of_this_case",
        dotnet = RuntimeInformation.FrameworkDescription,
        os = RuntimeInformation.OSDescription,
        architecture = RuntimeInformation.ProcessArchitecture.ToString(),
        torchSharp = typeof(Tensor).Assembly.GetName().Version?.ToString(),
        declaredLibtorchPackage = "2.10.0",
        cpu = Cpu.Value,
        cpuFeaturesDotNet = new
        {
            scope = "managed_intrinsic_availability_not_aten_dispatch",
            sse2 = Sse2.IsSupported, avx = Avx.IsSupported, avx2 = Avx2.IsSupported,
            avx512F = Avx512F.IsSupported, fma = Fma.IsSupported,
            armAdvSimd = System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported
        },
        threads = ObserveThreads(),
        requestedAtenCpuCapability = DescribeRequestedCapability(Environment.GetEnvironmentVariable("ATEN_CPU_CAPABILITY")),
        actualAtenCpuCapability = new
        {
            status = "unavailable", value = (string?)null,
            reason = "not_exposed_by_public_TorchSharp_API_CPUID_is_not_dispatch_evidence"
        },
        nativeLibraries = Libraries.Value
    };

    internal static RequestedCapability DescribeRequestedCapability(string? value) => value switch
    {
        null => new("unset", null),
        "default" or "avx2" or "avx512" or "vsx" or "zvector" or "sve256" => new("requested", value),
        _ => new("unrecognized_value_redacted", null)
    };

    private static object ObserveThreads()
    {
        try { return new { status = "available", intraOp = get_num_threads(), interOp = get_num_interop_threads() }; }
        catch (Exception error) { return new { status = "unavailable", errorType = error.GetType().Name }; }
    }

    internal static CpuEvidence ObserveCpu(bool supported, CpuIdQuery query)
    {
        var flags = new Dictionary<string, bool?>(StringComparer.Ordinal);
        if (!supported) return new("cpuid_unavailable_on_this_architecture", null, null, flags, null);
        try
        {
            var leaf0 = query(0, 0);
            string vendor = RegisterText([leaf0.Ebx, leaf0.Edx, leaf0.Ecx]);
            string? brand = null;
            var extended = query(unchecked((int)0x80000000U), 0);
            if ((uint)extended.Eax >= 0x80000004U)
            {
                int[] words = new int[12];
                for (int i = 0; i < 3; i++)
                {
                    var leaf = query(unchecked((int)(0x80000002U + (uint)i)), 0);
                    words[i * 4] = leaf.Eax; words[i * 4 + 1] = leaf.Ebx;
                    words[i * 4 + 2] = leaf.Ecx; words[i * 4 + 3] = leaf.Edx;
                }
                brand = RegisterText(words);
            }
            foreach (string name in new[] { "sse", "sse2", "sse3", "ssse3", "sse41", "sse42", "avx", "fma", "osxsave",
                "avx2", "avx512F", "avx512Dq", "avx512Bw", "avx512Vl" }) flags[name] = null;
            if ((uint)leaf0.Eax >= 1)
            {
                var leaf = query(1, 0);
                flags["sse"] = Bit(leaf.Edx, 25); flags["sse2"] = Bit(leaf.Edx, 26);
                flags["sse3"] = Bit(leaf.Ecx, 0); flags["ssse3"] = Bit(leaf.Ecx, 9);
                flags["sse41"] = Bit(leaf.Ecx, 19); flags["sse42"] = Bit(leaf.Ecx, 20);
                flags["avx"] = Bit(leaf.Ecx, 28); flags["fma"] = Bit(leaf.Ecx, 12); flags["osxsave"] = Bit(leaf.Ecx, 27);
            }
            if ((uint)leaf0.Eax >= 7)
            {
                var leaf = query(7, 0);
                flags["avx2"] = Bit(leaf.Ebx, 5); flags["avx512F"] = Bit(leaf.Ebx, 16);
                flags["avx512Dq"] = Bit(leaf.Ebx, 17); flags["avx512Bw"] = Bit(leaf.Ebx, 30); flags["avx512Vl"] = Bit(leaf.Ebx, 31);
            }
            return new("available", vendor, brand, flags, null);
        }
        catch (Exception error) { return new("observation_failed", null, null, flags, error.GetType().Name); }
    }

    private static bool Bit(int word, int bit) => ((uint)word & (1U << bit)) != 0;
    private static string RegisterText(ReadOnlySpan<int> words)
    {
        Span<byte> bytes = stackalloc byte[words.Length * 4];
        for (int i = 0; i < words.Length; i++) BinaryPrimitives.WriteInt32LittleEndian(bytes.Slice(i * 4, 4), words[i]);
        for (int i = 0; i < bytes.Length; i++) if (bytes[i] is < 32 or > 126) bytes[i] = 32;
        return Encoding.ASCII.GetString(bytes).Trim();
    }

    internal static Lazy<LibraryEvidence> LibraryCache(Func<LibraryEvidence> observe)
        => new(observe, LazyThreadSafetyMode.ExecutionAndPublication);

    private static IEnumerable<string> LoadedModulePaths()
    {
        using var process = Process.GetCurrentProcess();
        foreach (ProcessModule module in process.Modules) yield return module.FileName;
    }

    internal static LibraryEvidence ObserveLibraries(Func<IEnumerable<string>> enumerate, Func<string, Stream> open)
    {
        var libraries = new List<LibraryIdentity>();
        // Paths only exist as private cache keys. Distinct loaded instances remain
        // distinct records, even when basename or payload SHA happens to match.
        var files = new Dictionary<string, FileEvidence>(OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
        string? enumerationError = null;
        try
        {
            foreach (string path in enumerate())
            {
                string name = Path.GetFileName(path);
                if (!RelevantLibrary(name)) continue;
                if (!files.TryGetValue(path, out var file))
                {
                    try
                    {
                        using var stream = open(path);
                        long bytes = stream.Length;
                        string hash = Convert.ToHexStringLower(SHA256.HashData(stream));
                        file = new("available", bytes, hash, null);
                    }
                    catch (Exception error) { file = new("unavailable", null, null, error.GetType().Name); }
                    files.Add(path, file);
                }
                libraries.Add(new(libraries.Count + 1, name, file.status, file.bytes, file.sha256, file.errorType));
            }
        }
        catch (Exception error) { enumerationError = error.GetType().Name; }
        string status = libraries.Count == 0 ? "unavailable"
            : enumerationError is not null || libraries.Any(item => item.status != "available") ? "partial" : "available";
        return new(status, "cached_once_after_first_traced_unet_forward_in_process", libraries.AsReadOnly(),
            enumerationError ?? (libraries.Count == 0 ? "no_matching_loaded_modules" : null));
    }

    private static bool RelevantLibrary(string name)
    {
        if (name.Equals("TorchSharp.dll", StringComparison.OrdinalIgnoreCase)) return false;
        return new[] { "torch", "libtorch", "liblibtorchsharp", "c10", "libc10", "libgomp", "libomp", "libiomp", "iomp", "libshm" }
            .Any(prefix => name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase));
    }
}
