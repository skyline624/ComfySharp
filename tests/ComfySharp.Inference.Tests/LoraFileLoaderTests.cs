using System.Buffers.Binary;
using System.Text.Json;
using Xunit;
using static TorchSharp.torch;
namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LoraFileLoaderTests
{
    [Fact]
    public void Loaded_LoCon_and_DoRA_factors_match_the_existing_source_math_reference()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using var stream=GetType().Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.lora.reference.json")!;
        using var json=JsonDocument.Parse(stream);
        foreach(var row in json.RootElement.GetProperty("cases").EnumerateArray())
        {
            Value ConvertValue(JsonElement v)=>new(v.GetProperty("shape").EnumerateArray().Select(x=>x.GetInt64()).ToArray(),Floats(v.GetProperty("values")));
            var values=new Dictionary<string,Value>{{"layer.lora_up.weight",ConvertValue(row.GetProperty("up"))},{"layer.lora_down.weight",ConvertValue(row.GetProperty("down"))}};
            if(row.GetProperty("mid").ValueKind!=JsonValueKind.Null)values["layer.lora_mid.weight"]=ConvertValue(row.GetProperty("mid"));
            if(row.GetProperty("dora").ValueKind!=JsonValueKind.Null)values["layer.dora_scale"]=ConvertValue(row.GetProperty("dora"));
            if(row.GetProperty("alpha").ValueKind!=JsonValueKind.Null)values["layer.alpha"]=new([],[row.GetProperty("alpha").GetSingle()]);
            var weight=ConvertValue(row.GetProperty("weight"));string path=Write(values);
            try
            {
                using var scope=NewDisposeScope();using var file=new SafeTensorFile(path);
                var plan=LoraFileLoader.Inspect(file,[new("layer",new("model","weight",weight.Shape))]);
                using var adapters=LoraFileLoader.Load(file,plan,new Dictionary<string,double>{{"model",row.GetProperty("strength").GetDouble()}});
                using var result=adapters.Apply("model","weight",tensor(weight.Values).reshape(weight.Shape));
                var expected=Floats(row.GetProperty("output").GetProperty("values"));var actual=result.data<float>().ToArray();
                for(int i=0;i<expected.Length;i++)Assert.InRange(Math.Abs((double)actual[i]-expected[i]),0,1e-6+1e-6*Math.Abs(expected[i]));
            }
            finally{File.Delete(path);}
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
    private sealed record Value(long[] Shape,float[] Values,string DType="F32");
    private static string Write(IReadOnlyDictionary<string,Value> values)
    {
        var header=new Dictionary<string,object>();var payload=new List<byte>();
        foreach(var(name,value) in values)
        {
            int width=value.DType is "F16" or "BF16"?2:value.DType is "F64" or "I64"?8:4;
            byte[] bytes=new byte[value.Values.Length*width];
            for(int i=0;i<value.Values.Length;i++)
            {
                if(value.DType=="F32")BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i*4),value.Values[i]);
                else if(value.DType=="F64")BinaryPrimitives.WriteDoubleLittleEndian(bytes.AsSpan(i*8),value.Values[i]);
                else if(value.DType=="I64")BinaryPrimitives.WriteInt64LittleEndian(bytes.AsSpan(i*8),checked((long)value.Values[i]));
                else if(value.DType=="F16")BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i*2),BitConverter.HalfToUInt16Bits((Half)value.Values[i]));
                else if(value.DType=="BF16")BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i*2),(ushort)(BitConverter.SingleToUInt32Bits(value.Values[i])>>16));
            }
            header[name]=new{dtype=value.DType,shape=value.Shape,data_offsets=new[]{payload.Count,payload.Count+bytes.Length}};payload.AddRange(bytes);
        }
        byte[] json=JsonSerializer.SerializeToUtf8Bytes(header);string path=Path.GetTempFileName();
        using var stream=File.Create(path);Span<byte> length=stackalloc byte[8];BinaryPrimitives.WriteUInt64LittleEndian(length,(ulong)json.Length);
        stream.Write(length);stream.Write(json);stream.Write(payload.ToArray());return path;
    }
    private static Dictionary<string,Value> Basic()=>new(){["layer.lora_up.weight"]=new([2,1],[1,2]),["layer.lora_down.weight"]=new([1,3],[1,2,3])};
    private static LoraAlias Alias(string prefix="layer",string weight="weight")=>new(prefix,new("model",weight,Array.AsReadOnly(new long[]{2,3})));

    [Fact]
    public void Integer_alpha_is_read_as_a_scalar_without_reinterpreting_its_payload_bits()
    {
        NativeRuntimeBootstrap.Initialize();using var scope=NewDisposeScope();
        var values=Basic();values["layer.alpha"]=new([],[2],"I64");string path=Write(values);
        try
        {
            using var file=new SafeTensorFile(path);var plan=LoraFileLoader.Inspect(file,[Alias()]);using var adapters=LoraFileLoader.Load(file,plan);
            using var result=adapters.Apply("model","weight",ones(new long[]{2,3}));
            Assert.Equal(new[]{3f,5f,7f,5f,9f,13f},result.data<float>().ToArray());
        }
        finally{File.Delete(path);}
    }

    [Fact]
    public void Seven_formats_and_both_precedence_rules_match_actual_frozen_load_lora()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using var stream=GetType().Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.lora-loader.reference.json")!;
        using var json=JsonDocument.Parse(stream);
        foreach(var row in json.RootElement.GetProperty("cases").EnumerateArray())
        {
            var values=row.GetProperty("tensors").EnumerateObject().ToDictionary(p=>p.Name,p=>new Value(
                p.Value.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()).ToArray(),Floats(p.Value.GetProperty("values")),p.Value.GetProperty("dtype").GetString()!));
            string path=Write(values);
            try
            {
                using var scope=NewDisposeScope();using var file=new SafeTensorFile(path);
                var aliases=row.GetProperty("aliases").EnumerateArray().Select(v=>Alias(v.GetString()!)).ToArray();
                var plan=LoraFileLoader.Inspect(file,aliases,allowUnclaimedKeys:true);
                Assert.Equal(row.GetProperty("unclaimedKeys").EnumerateArray().Select(v=>v.GetString()),plan.UnclaimedKeys);
                Assert.Single(plan.Bindings);
                Assert.Equal(row.GetProperty("name").GetString()=="later-alias-wins"?new[]{"first"}:Array.Empty<string>(),plan.ShadowedPrefixes);
                using var adapters=LoraFileLoader.Load(file,plan,new Dictionary<string,double>{{"model",.5}});
                file.Dispose();
                var weight=tensor(Floats(row.GetProperty("weight").GetProperty("values"))).reshape(2,3);
                Assert.Throws<InvalidDataException>(()=>adapters.Apply("model","weight",weight.reshape(3,2)));
                using var result=adapters.Apply("model","weight",weight);
                var expected=Floats(row.GetProperty("output").GetProperty("values"));var actual=result.data<float>().ToArray();
                for(int i=0;i<expected.Length;i++)Assert.InRange(Math.Abs((double)actual[i]-expected[i]),0,1e-6+1e-6*Math.Abs(expected[i]));
            }
            finally{File.Delete(path);}
        }
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Fact]
    public void Inspection_rejects_incomplete_factors_unknown_keys_reshape_and_budget_without_payload_load()
    {
        void Check(Dictionary<string,Value> values,Action<SafeTensorFile> verify)
        {string path=Write(values);try{using var file=new SafeTensorFile(path);verify(file);}finally{File.Delete(path);}}
        var missing=Basic();missing.Remove("layer.lora_down.weight");Check(missing,f=>Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Inspect(f,[Alias()])));
        var other=Basic();other["unknown.lokr_w1"]=new([1,1],[1]);Check(other,f=>{
            Assert.Throws<NotSupportedException>(()=>LoraFileLoader.Inspect(f,[Alias()]));
            Assert.Equal(new[]{"unknown.lokr_w1"},LoraFileLoader.Inspect(f,[Alias()],true).UnclaimedKeys);});
        var reshape=Basic();reshape["layer.reshape_weight"]=new([2],[2,3]);Check(reshape,f=>Assert.Throws<NotSupportedException>(()=>LoraFileLoader.Inspect(f,[Alias()],true)));
        Check(Basic(),f=>{
            Assert.Throws<NotSupportedException>(()=>LoraFileLoader.Inspect(f,[Alias()],maxResidentFactorBytes:19));
            Assert.Equal(20,LoraFileLoader.Inspect(f,[Alias()],maxResidentFactorBytes:20).ResidentFactorBytes);
            Assert.Throws<ArgumentException>(()=>LoraFileLoader.Inspect(f,[Alias(),Alias()]));
            Assert.ThrowsAny<OperationCanceledException>(()=>LoraFileLoader.Inspect(f,[Alias()],cancellationToken:new(true)));});
    }

    [Fact]
    public void Loading_is_tied_to_reader_and_partial_failures_release_earlier_snapshots()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        var values=Basic();values["next.lora_up.weight"]=new([2,1],[1,1]);values["next.lora_down.weight"]=new([1,3],[1,1,1]);values["next.alpha"]=new([],[float.NaN]);
        string path=Write(values);
        try
        {
            using var file=new SafeTensorFile(path);using var another=new SafeTensorFile(path);
            var plan=LoraFileLoader.Inspect(file,[Alias(),Alias("next","second")]);
            Assert.Throws<ArgumentException>(()=>LoraFileLoader.Load(another,plan));
            Assert.ThrowsAny<OperationCanceledException>(()=>LoraFileLoader.Load(file,plan,cancellationToken:new(true)));
            Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Load(file,plan));
            Assert.Throws<ArgumentException>(()=>LoraFileLoader.Load(file,plan,new Dictionary<string,double>{{"missing",1}}));
        }
        finally{File.Delete(path);}
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Fact]
    public void Plan_snapshots_target_shape_and_loader_rejects_mismatched_factor_dimensions()
    {
        string path=Write(Basic());
        try
        {
            using var file=new SafeTensorFile(path);long[] shape=[2,3];
            var plan=LoraFileLoader.Inspect(file,[new("layer",new("model","weight",shape))]);shape[0]=100;
            Assert.Equal(new long[]{2,3},plan.Bindings[0].Target.Shape);
            Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Inspect(file,[new("layer",new("model","weight",shape))]));
            var declared=LoraFileLoader.Inspect(file,[Alias(),new("unmatched_clip",new("clip","weight",Array.AsReadOnly(new long[]{2,3})))]);
            using var adapters=LoraFileLoader.Load(file,declared,new Dictionary<string,double>{{"model",.5},{"clip",1}});
            Assert.Equal(new[]{"model","clip"},declared.Components);
        }
        finally{File.Delete(path);}
    }

    [Fact]
    public void Loaded_adapter_patches_actual_unet_and_survives_file_disposal()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;int threads=get_num_threads();set_num_threads(1);
        var values=new Dictionary<string,Value>{{"out.lora_up.weight",new([4,1],Enumerable.Repeat(.125f,4).ToArray())},
            {"out.lora_down.weight",new([1,32,3,3],Enumerable.Repeat(.01f,288).ToArray())}};
        string path=Write(values);
        try
        {
            using var scope=NewDisposeScope();using var file=new SafeTensorFile(path);
            var plan=LoraFileLoader.Inspect(file,[new("out",new("model","out.2.weight",Array.AsReadOnly(new long[]{4,32,3,3})))]);
            using var loaded=LoraFileLoader.Load(file,plan);file.Dispose();
            using var bank=SdSyntheticInputs.CreateUnet(new(32,16,SdAttentionHeadMode.FixedCount,4,false));using var original=new SdUnet(bank);
            Assert.Throws<ArgumentException>(()=>loaded.ApplyTo(original,component:"unknown"));
            using var patched=loaded.ApplyTo(original);loaded.Dispose();
            var x=ones(new long[]{1,4,4,5});var t=tensor(new[]{500f});var context=ones(new long[]{1,3,16});
            using var baseline=original.Forward(x,t,context);bank.Dispose();original.Dispose();
            using var prediction=patched.Forward(x,t,context);Assert.False(baseline.data<float>().ToArray().SequenceEqual(prediction.data<float>().ToArray()));
        }
        finally{File.Delete(path);set_num_threads(threads);}
        Assert.Equal(before,Tensor.TotalCount);
    }
    private static float[] Floats(JsonElement value)=>value.EnumerateArray().Select(v=>v.GetSingle()).ToArray();
}
