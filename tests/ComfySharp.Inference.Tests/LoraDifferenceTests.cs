using Xunit;
using System.Text.Json;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LoraDifferenceTests
{
    [Fact]
    public void Mixed_loader_and_application_follow_actual_frozen_source_for_all_five_precedence_cases()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using var corpus=JsonDocument.Parse(ClipReferenceTests.Resource("lora-differences.reference.json"));
        foreach(var row in corpus.RootElement.GetProperty("cases").EnumerateArray())
        {
            var values=row.GetProperty("tensors").EnumerateObject().ToDictionary(p=>p.Name,p=>new LoraFileLoaderTests.Value(
                p.Value.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()).ToArray(),Floats(p.Value)));
            string path=LoraFileLoaderTests.Write(values);
            try
            {
                using var scope=NewDisposeScope();using var file=new SafeTensorFile(path);
                var aliases=row.GetProperty("aliases").EnumerateArray().Select(v=>new LoraAlias(v.GetString()!,
                    new("model","layer.weight",row.GetProperty("targetShape").EnumerateArray().Select(x=>x.GetInt64()).ToArray()))).ToArray();
                var plan=LoraFileLoader.Inspect(file,aliases);
                using var loaded=LoraFileLoader.Load(file,plan,new Dictionary<string,double>{{"model",row.GetProperty("strength").GetDouble()}});
                Assert.Equal(row.GetProperty("outputs").EnumerateObject().Select(p=>p.Name).Order(),plan.Bindings.Select(b=>b.Target.Weight).Order());
                foreach(var output in row.GetProperty("outputs").EnumerateObject())
                {
                    var input=row.GetProperty("weights").GetProperty(output.Name);
                    using var actual=loaded.Apply("model",output.Name,tensor(Floats(input)).reshape(input.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()).ToArray()));
                    Assert.Equal(Floats(output.Value),actual.data<float>().ToArray());
                }
            }
            finally{File.Delete(path);}
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
    private static float[] Floats(JsonElement tensor)=>tensor.GetProperty("values").EnumerateArray().Select(v=>v.GetSingle()).ToArray();

    [Fact]
    public void All_686_targets_export_reload_and_freeze_independently_of_training_owners()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;int threads=get_num_threads();set_num_threads(1);
        string directory=Path.Combine(Path.GetTempPath(),"comfysharp-mixed-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(directory);
        string path=Path.Combine(directory,"adapter.safetensors");
        try
        {
            using var scope=NewDisposeScope();using var noGrad=no_grad();
            var config=new SdUnetConfig(32,16,SdAttentionHeadMode.FixedCount,4,false);
            using var set=new SdTrainableAdapterSet(config,2,317,CPU);
            foreach(var patch in set.Patches.Values)
                if(patch is TrainableDifferencePatch diff)diff.Difference.fill_(.001f);
                else if(patch is TrainableLoraPatch lora){lora.Down.fill_(.002f);lora.AlphaParameter!.fill_(.75f);}
            var frozen=set.Patches.ToDictionary(p=>p.Key,p=>p.Value switch
            {
                TrainableDifferencePatch diff=>diff.Snapshot(),
                TrainableLoraPatch lora=>lora.Snapshot(),
                _=>throw new InvalidOperationException()
            });
            try
            {
                LoraTrainingFile.SaveTargetsNew(path,set.Patches,maxFactorBytes:set.ParameterBytes);
                using var file=new SafeTensorFile(path);
                Assert.Equal(1250,file.Tensors.Count);
                Assert.All(file.Tensors.Values,t=>Assert.Equal("F32",t.DType));
                var plan=LoraFileLoader.Inspect(file,LoraModelAliases.ForUnet(config));
                Assert.Equal(686,plan.Bindings.Count);Assert.Empty(plan.UnclaimedKeys);
                using var loaded=LoraFileLoader.Load(file,plan);
                foreach(var patch in set.Patches.Values)foreach(var parameter in patch.Parameters)parameter.fill_(99);
                set.Dispose();file.Dispose();
                using var bank=SdSyntheticInputs.CreateUnet(config);using var model=new SdUnet(bank);
                using var expected=model.WithLora(frozen);using var actual=loaded.ApplyTo(model);loaded.Dispose();
                var x=ones([1,4,8,8]);var t=tensor(new[]{17.25f});var context=ones([1,3,16]);
                using var baseline=model.Forward(x,t,context);
                using var a=actual.Forward(x,t,context);using var e=expected.Forward(x,t,context);
                Assert.Equal(e.data<float>().ToArray(),a.data<float>().ToArray());
                Assert.False(baseline.data<float>().ToArray().SequenceEqual(a.data<float>().ToArray()));
            }
            finally{foreach(var patch in frozen.Values)patch.Dispose();}
        }
        finally{File.Delete(path);Directory.Delete(directory);set_num_threads(threads);}
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Fact]
    public void Invalid_differences_and_failed_exports_do_not_leave_files_or_native_owners()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            using var patch=new TrainableDifferencePatch(ones([2]));
            var targets=new Dictionary<string,TrainableDifferencePatch>{{"norm.weight",patch},{"norm.bias",patch}};
            string directory=Path.Combine(Path.GetTempPath(),"comfysharp-mixed-"+Guid.NewGuid().ToString("N"));Directory.CreateDirectory(directory);
            string path=Path.Combine(directory,"adapter.safetensors");
            try
            {
                Assert.Throws<NotSupportedException>(()=>LoraTrainingFile.SaveTargetsNew(path,targets,maxFactorBytes:15));
                Assert.ThrowsAny<OperationCanceledException>(()=>LoraTrainingFile.SaveTargetsNew(path,targets,cancellationToken:new(true)));
                using(var noGrad=no_grad())patch.Difference.fill_(float.NaN);
                Assert.Throws<ArithmeticException>(()=>LoraTrainingFile.SaveTargetsNew(path,targets));
                Assert.Empty(Directory.GetFiles(directory));
                using(var noGrad=no_grad())patch.Difference.fill_(1);
                LoraTrainingFile.SaveTargetsNew(path,targets);
                byte[] original=File.ReadAllBytes(path);
                Assert.Throws<IOException>(()=>LoraTrainingFile.SaveTargetsNew(path,targets));
                Assert.Equal(original,File.ReadAllBytes(path));
            }
            finally{File.Delete(path);Directory.Delete(directory);}
        }
        foreach(var values in new[]{new float[]{1},new float[]{float.NaN,1}})
        {
            string path=LoraFileLoaderTests.Write(new Dictionary<string,LoraFileLoaderTests.Value>{{"layer.diff",new([values.Length],values)}});
            try
            {
                using var file=new SafeTensorFile(path);
                if(values.Length==1)Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Inspect(file,[new("layer",new("model","layer.weight",[2]))]));
                else
                {
                    var plan=LoraFileLoader.Inspect(file,[new("layer",new("model","layer.weight",[2]))]);
                    Assert.Throws<InvalidDataException>(()=>LoraFileLoader.Load(file,plan));
                }
            }
            finally{File.Delete(path);}
        }
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Fact]
    public void Difference_overrides_factors_and_bias_is_loaded_independently_with_component_strength()
    {
        NativeRuntimeBootstrap.Initialize(); long before = Tensor.TotalCount;
        string path = LoraFileLoaderTests.Write(new Dictionary<string,LoraFileLoaderTests.Value>()
        {
            ["layer.lora_up.weight"] = new([2,1], [9,9]),
            ["layer.lora_down.weight"] = new([1,3], [9,9,9]),
            ["layer.diff"] = new([2,3], [1,2,3,4,5,6]),
            ["layer.diff_b"] = new([2], [2,4])
        });
        try
        {
            using var scope = NewDisposeScope(); using var file = new SafeTensorFile(path);
            var plan = LoraFileLoader.Inspect(file, [new("layer", new("model", "layer.weight", [2,3]))]);
            Assert.Empty(plan.UnclaimedKeys);
            Assert.Equal(32, plan.ResidentFactorBytes);
            using var loaded = LoraFileLoader.Load(file, plan, new Dictionary<string,double>{{"model",.5}});
            file.Dispose();
            using var weight = loaded.Apply("model", "layer.weight", ones([2,3]));
            using var bias = loaded.Apply("model", "layer.bias", ones([2]));
            Assert.Equal(new[]{1.5f,2,2.5f,3,3.5f,4}, weight.data<float>().ToArray());
            Assert.Equal(new[]{2f,3}, bias.data<float>().ToArray());
        }
        finally { File.Delete(path); }
        Assert.Equal(before, Tensor.TotalCount);
    }
}
