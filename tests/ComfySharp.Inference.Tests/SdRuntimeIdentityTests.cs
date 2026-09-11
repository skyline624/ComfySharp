using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using Xunit;

namespace ComfySharp.Inference.Tests;

public sealed class SdRuntimeIdentityTests
{
    [Fact]
    public void CpuIdDecodesVendorBrandAndHardwareFlagsWithoutClaimingDispatch()
    {
        byte[] brand = Encoding.ASCII.GetBytes("Diagnostic CPU".PadRight(48));
        (int, int, int, int) Query(int leaf, int subleaf)
        {
            Assert.Equal(0, subleaf);
            return unchecked((uint)leaf) switch
            {
                0 => (7, Word("Genu"), Word("ntel"), Word("ineI")),
                1 => (0, 0, (1 << 28) | (1 << 27) | (1 << 12), 1 << 26),
                7 => (0, unchecked((int)((1U << 5) | (1U << 16) | (1U << 31))), 0, 0),
                0x80000000 => (unchecked((int)0x80000004), 0, 0, 0),
                >= 0x80000002 and <= 0x80000004 => BrandLeaf(checked((int)(unchecked((uint)leaf) - 0x80000002))),
                _ => throw new InvalidOperationException()
            };
        }
        (int, int, int, int) BrandLeaf(int index)
            => (Read(index * 16), Read(index * 16 + 4), Read(index * 16 + 8), Read(index * 16 + 12));
        int Read(int offset) => BinaryPrimitives.ReadInt32LittleEndian(brand.AsSpan(offset, 4));
        var observed = SdRuntimeIdentity.ObserveCpu(true, Query);
        Assert.Equal("available", observed.status);
        Assert.Equal("GenuineIntel", observed.vendor);
        Assert.Equal("Diagnostic CPU", observed.brand);
        Assert.True(observed.hardwareFeatures["avx2"]);
        Assert.True(observed.hardwareFeatures["avx512F"]);
        Assert.True(observed.hardwareFeatures["avx512Vl"]);
        Assert.False(observed.hardwareFeatures["avx512Bw"]);
        Assert.Equal("cpuid_unavailable_on_this_architecture",
            SdRuntimeIdentity.ObserveCpu(false, (_, _) => throw new Exception()).status);
        var requested = SdRuntimeIdentity.DescribeRequestedCapability("Z:/private/person");
        Assert.Equal("unrecognized_value_redacted", requested.status);
        Assert.Null(requested.value);
        Assert.DoesNotContain("private", JsonSerializer.Serialize(requested));
    }

    [Fact]
    public void LibraryObservationPreservesDuplicateBasenamesAndRedactsFailedPaths()
    {
        string first = Path.Combine(Path.GetTempPath(), "private-a", "torch_cpu.dll");
        string second = Path.Combine(Path.GetTempPath(), "private-b", "torch_cpu.dll");
        string failed = Path.Combine(Path.GetTempPath(), "private-c", "libiomp5md.dll");
        var reads = new Dictionary<string, int>();
        var observed = SdRuntimeIdentity.ObserveLibraries(() => [first, second, first, failed], path =>
        {
            reads[path] = reads.GetValueOrDefault(path) + 1;
            if (path == failed) throw new IOException("Private path: " + path);
            return new MemoryStream(path == first ? [1, 2, 3] : [4, 5, 6]);
        });
        Assert.Equal("partial", observed.status);
        Assert.Equal(4, observed.libraries.Count);
        Assert.Equal(new[] { 1, 2, 3, 4 }, observed.libraries.Select(item => item.instance));
        Assert.Equal("torch_cpu.dll", observed.libraries[0].name);
        Assert.Equal(observed.libraries[0].name, observed.libraries[1].name);
        Assert.NotEqual(observed.libraries[0].sha256, observed.libraries[1].sha256);
        Assert.Equal(observed.libraries[0].sha256, observed.libraries[2].sha256);
        Assert.Equal("IOException", observed.libraries[3].errorType);
        Assert.Null(observed.libraries[3].sha256);
        Assert.All(reads.Values, count => Assert.Equal(1, count));
        Assert.DoesNotContain("private", JsonSerializer.Serialize(observed));
    }

    [Fact]
    public void LibraryCacheIsDeferredAndObservationFailureRemainsExplicit()
    {
        int calls = 0;
        var cache = SdRuntimeIdentity.LibraryCache(() =>
        {
            calls++;
            return SdRuntimeIdentity.ObserveLibraries(() => throw new PlatformNotSupportedException("private path"),
                _ => throw new InvalidOperationException());
        });
        Assert.Equal(0, calls);
        Assert.Equal("unavailable", cache.Value.status);
        Assert.Equal("PlatformNotSupportedException", cache.Value.errorType);
        Assert.Empty(cache.Value.libraries);
        Assert.Same(cache.Value, cache.Value);
        Assert.Equal(1, calls);
        Assert.DoesNotContain("private", JsonSerializer.Serialize(cache.Value));
    }

    private static int Word(string text) => BinaryPrimitives.ReadInt32LittleEndian(Encoding.ASCII.GetBytes(text));
}
