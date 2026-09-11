using System.Buffers.Binary;
using System.Text;
using ComfySharp.Inference;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

public sealed class InferenceTests
{
    private static string Fixture(string header, byte[]? data = null)
    {
        string path = Path.GetTempFileName();
        using var file = File.Create(path);
        var json = Encoding.UTF8.GetBytes(header);
        Span<byte> prefix = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)json.Length);
        file.Write(prefix); file.Write(json); file.Write(data ?? new byte[4]);
        return path;
    }

    [Theory]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"a\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[4,4]}}")]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}")]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}")]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,4]}}")]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[9223372036854775807,2],\"data_offsets\":[0,4]}}")]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}")]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}")]
    [InlineData("{\"a\":{\"dtype\":\"PICKLE\",\"shape\":[1],\"data_offsets\":[0,4]}}")]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[-1,3]}}")]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[4,4]}}")]
    [InlineData("{\"a\":{\"dtype\":\"F32\",\"shape\":[1]}}")]
    [InlineData("{\"__metadata__\":{\"author\":1}}")]
    [InlineData("{\"__metadata__\":{\"a\":\"1\",\"a\":\"2\"}}")]
    [InlineData("{}")]
    [InlineData("[]")]
    [InlineData("{invalid}")]
    public void RejectsMalformedAndUnindexedFiles(string header)
    {
        string path = Fixture(header);
        try { Assert.Throws<InvalidDataException>(() => new SafeTensorFile(path)); }
        finally { File.Delete(path); }
    }

    [Fact]
    public void RejectsHeaderBombTruncationAndLimits()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(path, Enumerable.Repeat((byte)255, 8).ToArray());
            Assert.Throws<InvalidDataException>(() => new SafeTensorFile(path));
            File.WriteAllBytes(path, new byte[7]);
            Assert.Throws<InvalidDataException>(() => new SafeTensorFile(path));
        }
        finally { File.Delete(path); }
        path = Fixture("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}");
        try
        {
            Assert.Throws<InvalidDataException>(() => new SafeTensorFile(path, new(MaxTensorBytes: 3)));
            Assert.Throws<InvalidDataException>(() => new SafeTensorFile(path, new(MaxHeaderBytes: 8)));
            Assert.Throws<InvalidDataException>(() => new SafeTensorFile(path, new(MaxRank: 0)));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Float32TensorOwnsStorageAfterFileDisposal()
    {
        var bytes = new byte[8];
        BinaryPrimitives.WriteSingleLittleEndian(bytes, 1.25f);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(4), -3.5f);
        string path = Fixture("{\"__metadata__\":{\"format\":\"pt\"},\"a\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[0,8]}}", bytes);
        try
        {
            using var scope = NewDisposeScope();
            var file = new SafeTensorFile(path);
            Assert.Equal("pt", file.Metadata["format"]);
            var value = file.ReadFloat32("a");
            file.Dispose();
            File.Delete(path);
            Assert.Equal(new long[] { 1, 2 }, value.shape);
            Assert.Equal(new float[] { 1.25f, -3.5f }, value.data<float>().ToArray());
            Assert.Throws<ObjectDisposedException>(() => file.ReadFloat32("a"));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void EmptyAndScalarTensorsAreValidAndCancellationIsHonored()
    {
        string path = Fixture("{\"empty\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[0,0]},\"scalar\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]}}");
        try
        {
            using var scope = NewDisposeScope();
            using var file = new SafeTensorFile(path);
            Assert.Equal(0, file.ReadFloat32("empty").numel());
            Assert.Equal(0f, file.ReadFloat32("scalar").item<float>());
            Assert.Throws<OperationCanceledException>(() => file.ReadFloat32("scalar", new(true)));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void CancellationDuringMaterializationDisposesNativeResult()
    {
        string path = Fixture("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}");
        try
        {
            using var file = new SafeTensorFile(path);
            using var cancellation = new CancellationTokenSource();
            Tensor? materialized = null;
            Assert.Throws<OperationCanceledException>(() => file.ReadFloat32("a", cancellation.Token, (values, shape) =>
            {
                materialized = tensor(values).reshape(shape);
                cancellation.Cancel();
                return materialized;
            }));
            Assert.NotNull(materialized);
            Assert.True(materialized.IsInvalid);
            using var next = file.ReadFloat32("a");
            Assert.Equal(0f, next.item<float>());
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void NativeRngIsRepeatableAndSeedSensitive()
    {
        using var scope = NewDisposeScope();
        var a = NativeMath.CpuNoise(new long[] { 32 }, 123);
        var b = NativeMath.CpuNoise(new long[] { 32 }, 123);
        var c = NativeMath.CpuNoise(new long[] { 32 }, 124);
        Assert.Equal(a.data<float>().ToArray(), b.data<float>().ToArray());
        Assert.False(equal(a, c).all().item<bool>());
        Assert.Throws<OperationCanceledException>(() => NativeMath.CpuNoise(new long[] { 1 }, 1, new(true)));
    }

    [Fact]
    public void EulerHasKnownValuesAndDoesNotMutateInputs()
    {
        using var scope = NewDisposeScope();
        var x = tensor(new float[] { 4, 8 }); var denoised = tensor(new float[] { 2, 4 });
        Assert.Equal(new float[] { 3, 6 }, NativeMath.EulerStep(x, denoised, 2, 1).data<float>().ToArray());
        Assert.Equal(new float[] { 2, 4 }, NativeMath.EulerStep(x, denoised, 2, 0).data<float>().ToArray());
        Assert.Equal(new float[] { 4, 8 }, x.data<float>().ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() => NativeMath.EulerStep(x, denoised, 0, 0));
        Assert.Throws<ArgumentException>(() => NativeMath.EulerStep(x, ones(1), 2, 1));
        Assert.Throws<OperationCanceledException>(() => NativeMath.EulerStep(x, denoised, 2, 1, new(true)));
    }

    [Fact]
    public void NativeGradientOptimizerAndViewLifetime()
    {
        using var scope = NewDisposeScope();
        var p = nn.Parameter(tensor(new float[] { 2 }));
        using var optimizer = optim.SGD(new[] { p }, 0.1);
        optimizer.zero_grad(); p.square().sum().backward();
        Assert.Equal(4f, p.grad!.item<float>());
        optimizer.step(); Assert.Equal(1.6f, p.item<float>(), 5);
        var storageOwner = tensor(new float[] { 1, 2, 3, 4 });
        var view = storageOwner.reshape(2, 2); storageOwner.Dispose();
        Assert.Equal(10f, view.sum().item<float>());
        Assert.Equal(16f, nn.functional.conv2d(ones(new long[] { 1, 1, 3, 3 }), ones(new long[] { 1, 1, 2, 2 })).sum().item<float>());
        var q = ones(new long[] { 1, 1, 2, 2 });
        Assert.Equal(4f, nn.functional.scaled_dot_product_attention(q, q, q).sum().item<float>(), 5);
    }
}
