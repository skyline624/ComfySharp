using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using ComfySharp.Inference;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

public sealed class SafeTensorDTypeTests
{
    private static string Fixture(string dtype, long[] shape, byte[] bytes)
    {
        string path = Path.GetTempFileName();
        byte[] header = Encoding.UTF8.GetBytes($"{{\"value\":{{\"dtype\":\"{dtype}\",\"shape\":[{string.Join(',', shape)}],\"data_offsets\":[0,{bytes.Length}]}}}}");
        using var file = File.Create(path);
        Span<byte> prefix = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)header.Length);
        file.Write(prefix); file.Write(header); file.Write(bytes);
        return path;
    }

    public static TheoryData<string, ScalarType, string> BitPatterns => new()
    {
        // Each floating-point case includes +0, -0, the smallest subnormal, +/- infinity and a NaN payload.
        { "F16", ScalarType.Float16, "000000800100007C00FC557E" },
        { "BF16", ScalarType.BFloat16, "000000800100807F80FF457F" },
        { "F32", ScalarType.Float32, "0000000000000080010000000000807F000080FF4523C17F" },
        { "F64", ScalarType.Float64, "000000000000000000000000000000800100000000000000000000000000F07F000000000000F0FF452300000000F87F" },
        { "U8", ScalarType.Byte, "00017FFF" },
        { "I8", ScalarType.Int8, "00017F80FF" },
        { "I16", ScalarType.Int16, "000001003412FF7F0080FFFF" },
        { "I32", ScalarType.Int32, "000000000100000078563412FFFFFF7F00000080FFFFFFFF" },
        { "I64", ScalarType.Int64, "00000000000000000100000000000000EFCDAB8967452301FFFFFFFFFFFFFF7F0000000000000080FFFFFFFFFFFFFFFF" },
        { "BOOL", ScalarType.Bool, "0001000101" }
    };

    [Theory]
    [MemberData(nameof(BitPatterns))]
    public void CopiesExactNativeBitsAndOwnsStorageAfterFileAndGc(string dtype, ScalarType expectedType, string hex)
    {
        NativeRuntimeBootstrap.Initialize();
        byte[] original = Convert.FromHexString(hex);
        int width = dtype is "F64" or "I64" ? 8 : dtype is "F32" or "I32" ? 4 : dtype is "F16" or "BF16" or "I16" ? 2 : 1;
        string path = Fixture(dtype, [1, original.Length / width], original);
        try
        {
            using var scope = NewDisposeScope();
            Tensor result;
            using (var file = new SafeTensorFile(path)) result = file.ReadTensor("value");
            File.Delete(path);
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
            GC.WaitForPendingFinalizers();
            Assert.Equal(expectedType, result.dtype);
            Assert.Equal(DeviceType.CPU, result.device_type);
            Assert.Equal(new long[] { 1, original.Length / width }, result.shape);
            if (!BitConverter.IsLittleEndian)
                for (int offset = 0; offset < original.Length; offset += width) Array.Reverse(original, offset, width);
            Assert.Equal(original, result.bytes.ToArray());
        }
        finally { File.Delete(path); }
    }

    [Theory]
    [InlineData("F16", 2)]
    [InlineData("BF16", 2)]
    [InlineData("F32", 4)]
    [InlineData("F64", 8)]
    [InlineData("I64", 8)]
    [InlineData("BOOL", 1)]
    public void EmptyAndScalarShapesKeepTheirRank(string dtype, int width)
    {
        NativeRuntimeBootstrap.Initialize();
        string empty = Fixture(dtype, [2, 0, 3], []);
        string scalar = Fixture(dtype, [], new byte[width]);
        try
        {
            using var emptyFile = new SafeTensorFile(empty);
            using var scalarFile = new SafeTensorFile(scalar);
            using var emptyTensor = emptyFile.ReadTensor("value");
            using var scalarTensor = scalarFile.ReadTensor("value");
            Assert.Equal(new long[] { 2, 0, 3 }, emptyTensor.shape);
            Assert.Equal(0, emptyTensor.numel());
            Assert.Empty(scalarTensor.shape);
            Assert.Equal(1, scalarTensor.numel());
            Assert.Equal(new byte[width], scalarTensor.bytes.ToArray());
        }
        finally { File.Delete(empty); File.Delete(scalar); }
    }

    [Fact]
    public void LittleEndianBytesProduceKnownNumericValues()
    {
        NativeRuntimeBootstrap.Initialize();
        var cases = new[] { ("F16", "003C00C1", new[] { 1d, -2.5d }), ("BF16", "803F20C0", new[] { 1d, -2.5d }),
            ("F32", "0000803F000020C0", new[] { 1d, -2.5d }), ("F64", "000000000000F03F00000000000004C0", new[] { 1d, -2.5d }),
            ("I16", "3412FEFF", new[] { 4660d, -2d }), ("I32", "78563412FEFFFFFF", new[] { 305419896d, -2d }),
            ("I64", "7856341200000000FEFFFFFFFFFFFFFF", new[] { 305419896d, -2d }) };
        foreach (var (dtype, hex, expected) in cases)
        {
            string path = Fixture(dtype, [2], Convert.FromHexString(hex));
            try
            {
                using var file = new SafeTensorFile(path);
                using var result = file.ReadTensor("value");
                using var values = result.to_type(ScalarType.Float64);
                Assert.Equal(expected, values.data<double>().ToArray());
            }
            finally { File.Delete(path); }
        }
    }

    [Theory]
    [InlineData("U16", 2)]
    [InlineData("U32", 4)]
    [InlineData("U64", 8)]
    public void UnsignedApiGapsRemainInspectableWithExplicitErrors(string dtype, int width)
    {
        string path = Fixture(dtype, [1], new byte[width]);
        try
        {
            using var file = new SafeTensorFile(path);
            Assert.Equal(dtype, file.Tensors["value"].DType);
            Assert.False(SafeTensorFile.SupportsTensorDType(dtype));
            bool allocated = false;
            var error = Assert.Throws<NotSupportedException>(() => file.ReadTensor("value", default,
                (_, _) => { allocated = true; throw new Exception("Must not allocate."); }));
            Assert.Contains(dtype, error.Message);
            Assert.False(allocated);
            Assert.Throws<NotSupportedException>(() => file.ReadFloat32("value"));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void CancelBeforeAllocationDoesNotEnterNativeBoundary()
    {
        string path = Fixture("BF16", [1], new byte[2]);
        try
        {
            using var file = new SafeTensorFile(path);
            bool allocated = false;
            Assert.Throws<OperationCanceledException>(() => file.ReadTensor("value", new(true),
                (_, _) => { allocated = true; throw new Exception("Must not allocate."); }));
            Assert.False(allocated);
        }
        finally { File.Delete(path); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CancellationDisposesAllocatedTensorAndAllowsSubsequentRead(bool cancelAfterFirstChunk)
    {
        NativeRuntimeBootstrap.Initialize();
        byte[] data = new byte[1024 * 1024 + 8];
        data[^1] = 42;
        string path = Fixture("U8", [data.Length], data);
        try
        {
            using var file = new SafeTensorFile(path);
            using var cancellation = new CancellationTokenSource();
            Tensor? allocated = null;
            int chunks = 0;
            Assert.Throws<OperationCanceledException>(() => file.ReadTensor("value", cancellation.Token, (shape, dtype) =>
            {
                allocated = empty(shape, dtype: dtype, device: CPU);
                if (!cancelAfterFirstChunk) cancellation.Cancel();
                return allocated;
            }, () => { chunks++; cancellation.Cancel(); }));
            Assert.NotNull(allocated);
            Assert.True(allocated.IsInvalid);
            Assert.Equal(cancelAfterFirstChunk ? 1 : 0, chunks);
            using var result = file.ReadTensor("value");
            Assert.Equal(data, result.bytes.ToArray());
            file.Dispose();
            Assert.Throws<ObjectDisposedException>(() => file.ReadTensor("value"));
        }
        finally { File.Delete(path); }
    }

    [Theory]
    [InlineData(1, "0102030405060708")]
    [InlineData(2, "0201040306050807")]
    [InlineData(4, "0403020108070605")]
    [InlineData(8, "0807060504030201")]
    public void NativeByteOrderConversionReversesEachElementOnly(int width, string expectedHex)
    {
        byte[] bytes = [1, 2, 3, 4, 5, 6, 7, 8];
        SafeTensorFile.NormalizeByteOrder(bytes, width, isLittleEndian: true);
        Assert.Equal(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 }, bytes);
        SafeTensorFile.NormalizeByteOrder(bytes, width, isLittleEndian: false);
        Assert.Equal(Convert.FromHexString(expectedHex), bytes);
    }

    [Fact]
    public void HashMatchesFileBytesAndCancellationKeepsReaderUsable()
    {
        string path = Fixture("BF16", [2], Convert.FromHexString("803F20C0"));
        try
        {
            byte[] bytes = File.ReadAllBytes(path);
            using var file = new SafeTensorFile(path);
            Assert.Equal(bytes.Length, file.FileSizeBytes);
            Assert.Throws<OperationCanceledException>(() => file.ComputeSha256(new(true)));
            Assert.Equal(Convert.ToHexStringLower(SHA256.HashData(bytes)), file.ComputeSha256());
            Assert.Equal("BF16", file.Tensors["value"].DType);
            file.Dispose();
            Assert.Throws<ObjectDisposedException>(() => file.ComputeSha256());
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void UnixPathReplacementDoesNotChangeInspectedFileOrHash()
    {
        // Windows FileShare.Read prevents rename/deletion; Unix open handles survive path replacement.
        if (OperatingSystem.IsWindows()) return;
        string path = Fixture("BF16", [1], Convert.FromHexString("803F"));
        string replacement = Fixture("F32", [2], new byte[8]);
        try
        {
            byte[] original = File.ReadAllBytes(path);
            using var file = new SafeTensorFile(path);
            File.Move(replacement, path, overwrite: true);
            Assert.Equal("BF16", file.Tensors["value"].DType);
            Assert.Equal(original.Length, file.FileSizeBytes);
            Assert.Equal(Convert.ToHexStringLower(SHA256.HashData(original)), file.ComputeSha256());
            Assert.NotEqual(Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path))), file.ComputeSha256());
        }
        finally { File.Delete(path); File.Delete(replacement); }
    }
}
