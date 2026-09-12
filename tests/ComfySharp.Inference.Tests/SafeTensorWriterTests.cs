using System.Buffers.Binary;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class SafeTensorWriterTests
{
    private sealed class CallbackStream(Action callback) : MemoryStream
    {
        private bool invoked;
        public override void Write(ReadOnlySpan<byte> buffer)
        {
            if (!invoked) { invoked = true; callback(); }
            base.Write(buffer);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Snapshot_is_independent_and_cancellation_releases_native_copies(bool cancel)
    {
        NativeRuntimeBootstrap.Initialize();
        long before = Tensor.TotalCount;
        using (var scope = NewDisposeScope())
        using (var cancellation = new CancellationTokenSource())
        {
            var input = tensor(new float[] { 1, 2, 3 });
            using var output = new CallbackStream(() =>
            {
                input.fill_(9);
                if (cancel) cancellation.Cancel();
            });
            var state = new Dictionary<string, Tensor> { ["weight"] = input };
            long borrowed = Tensor.TotalCount;
            if (cancel)
                Assert.Throws<OperationCanceledException>(() => SafeTensorWriter.Write(output, state, cancellationToken: cancellation.Token));
            else
            {
                SafeTensorWriter.Write(output, state);
                byte[] bytes = output.ToArray();
                int payload = 8 + checked((int)BinaryPrimitives.ReadUInt64LittleEndian(bytes));
                Assert.Equal(new float[] { 1, 2, 3 }, new[] { BitConverter.ToSingle(bytes, payload), BitConverter.ToSingle(bytes, payload + 4), BitConverter.ToSingle(bytes, payload + 8) });
            }
            Assert.Equal(borrowed, Tensor.TotalCount);
            Assert.True(output.CanWrite);
            Assert.Equal(new float[] { 9, 9, 9 }, input.data<float>().ToArray());
        }
        Assert.Equal(before, Tensor.TotalCount);
    }

    [Theory]
    [InlineData(ScalarType.Bool,"BOOL")] [InlineData(ScalarType.Byte,"U8")] [InlineData(ScalarType.Int8,"I8")]
    [InlineData(ScalarType.Int16,"I16")] [InlineData(ScalarType.Int32,"I32")] [InlineData(ScalarType.Int64,"I64")]
    [InlineData(ScalarType.Float16,"F16")] [InlineData(ScalarType.BFloat16,"BF16")]
    [InlineData(ScalarType.Float32,"F32")] [InlineData(ScalarType.Float64,"F64")]
    public void All_reader_dtypes_preserve_shape_and_exact_native_bytes(ScalarType dtype, string expected)
    {
        NativeRuntimeBootstrap.Initialize(); long before=Tensor.TotalCount;
        string path=Path.Combine(Path.GetTempPath(),"comfysharp-state-"+Guid.NewGuid().ToString("N")+".safetensors");
        try
        {
            using var scope=NewDisposeScope(); var value=tensor(new float[]{0,1,1,0}).to_type(dtype).reshape(2,2);
            byte[] original=value.bytes.ToArray();
            using(var stream=File.Create(path))SafeTensorWriter.Write(stream,new Dictionary<string,Tensor>{{"adapter.weight",value}});
            byte[] encoded=File.ReadAllBytes(path); int headerLength=checked((int)BinaryPrimitives.ReadUInt64LittleEndian(encoded));
            using var json=JsonDocument.Parse(encoded.AsMemory(8,headerLength));
            Assert.Equal(expected,json.RootElement.GetProperty("adapter.weight").GetProperty("dtype").GetString());
            Assert.Equal(original,encoded[(8+headerLength)..]);
            value.fill_(0); using var reader=new SafeTensorFile(path); using var copy=reader.ReadTensor("adapter.weight");
            Assert.Equal(new long[]{2,2},copy.shape); Assert.Equal(dtype,copy.dtype); Assert.Equal(original,copy.bytes.ToArray());
        }
        finally { if(File.Exists(path))File.Delete(path); }
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Fact]
    public void Empty_state_empty_tensor_scalars_and_nonfinite_bits_are_serializable()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            using var output=new MemoryStream();SafeTensorWriter.Write(output,new Dictionary<string,Tensor>());
            Assert.Equal(16,output.Length);output.SetLength(0);
            var emptyTensor=empty(new long[]{0,2});var scalar=tensor(double.PositiveInfinity);var nan=tensor(float.NaN);
            SafeTensorWriter.Write(output,new Dictionary<string,Tensor>{{"empty",emptyTensor},{"scalar",scalar},{"nan",nan}});
            byte[] bytes=output.ToArray();int length=(int)BinaryPrimitives.ReadUInt64LittleEndian(bytes);
            using var json=JsonDocument.Parse(bytes.AsMemory(8,length));
            Assert.Empty(json.RootElement.GetProperty("scalar").GetProperty("shape").EnumerateArray());
            var offset=json.RootElement.GetProperty("scalar").GetProperty("data_offsets")[0].GetInt32();
            Assert.Equal(double.PositiveInfinity,BitConverter.ToDouble(bytes,8+length+offset));
            Assert.Equal(0,emptyTensor.numel());
        }
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Fact]
    public void Admission_and_cancellation_precede_destination_writes_and_leave_inputs_owned()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            using var output=new MemoryStream();var input=ones(2,3);var strided=input.transpose(0,1);
            Assert.Throws<ArgumentException>(()=>SafeTensorWriter.Write(output,new Dictionary<string,Tensor>{{"view",strided}}));
            Assert.Throws<NotSupportedException>(()=>SafeTensorWriter.Write(output,new Dictionary<string,Tensor>{{"value",input}},maxSnapshotBytes:1));
            Assert.Throws<ArgumentException>(()=>SafeTensorWriter.Write(output,new Dictionary<string,Tensor>{{"__metadata__",input}}));
            Assert.Throws<OperationCanceledException>(()=>SafeTensorWriter.Write(output,new Dictionary<string,Tensor>{{"value",input}},cancellationToken:new(true)));
            Assert.Equal(0,output.Length);Assert.Equal(6,input.sum().item<float>());
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
}
