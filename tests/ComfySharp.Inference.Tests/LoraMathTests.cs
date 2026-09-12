using Xunit;
using System.Text.Json;
using static TorchSharp.torch;
namespace ComfySharp.Inference.Tests;

[Collection("Classical VAE")]
public sealed class LoraMathTests
{
    [Fact]
    public void LoRA_LoCon_DoRA_and_source_training_gradients_match_frozen_reference()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using var stream=GetType().Assembly.GetManifestResourceStream("ComfySharp.Inference.Tests.Fixtures.lora.reference.json")!;
        using var json=JsonDocument.Parse(stream);
        foreach(var row in json.RootElement.GetProperty("cases").EnumerateArray())
        {
            using var scope=NewDisposeScope();
            var weight=Read(row.GetProperty("weight"));var up=Read(row.GetProperty("up"));var down=Read(row.GetProperty("down"));
            var mid=Optional(row.GetProperty("mid"));var dora=Optional(row.GetProperty("dora"));
            double? alpha=row.GetProperty("alpha").ValueKind==JsonValueKind.Null?null:row.GetProperty("alpha").GetDouble();
            using var result=LoraMath.Apply(weight,up,down,row.GetProperty("strength").GetDouble(),alpha,mid,dora);
            Compare(result,row.GetProperty("output"));Compare(weight,row.GetProperty("weight"));Compare(up,row.GetProperty("up"));Compare(down,row.GetProperty("down"));
        }
        using(var scope=NewDisposeScope())
        using(var grad=set_grad_enabled(true))
        {
            var row=json.RootElement.GetProperty("training");
            var weight=Read(row.GetProperty("weight"));
            var up=Read(row.GetProperty("up")).detach().requires_grad_();var down=Read(row.GetProperty("down")).detach().requires_grad_();
            using var result=LoraMath.Apply(weight,up,down,alpha:row.GetProperty("alpha").GetDouble());
            Compare(result,row.GetProperty("output"));var loss=result.square().mean();loss.backward();
            using var upGradient=up.grad!;using var downGradient=down.grad!;
            Compare(upGradient,row.GetProperty("upGradient"));Compare(downGradient,row.GetProperty("downGradient"));
            Assert.InRange(Math.Abs(loss.item<float>()-row.GetProperty("loss").GetDouble()),0,1e-6);
            using(var noGrad=no_grad())
            {up.sub_(upGradient*row.GetProperty("learningRate").GetDouble());down.sub_(downGradient*row.GetProperty("learningRate").GetDouble());}
            Compare(up,row.GetProperty("upUpdated"));Compare(down,row.GetProperty("downUpdated"));
            using var updated=LoraMath.Apply(weight,up,down,alpha:row.GetProperty("alpha").GetDouble());
            Compare(updated,row.GetProperty("outputUpdated"));Assert.Null(weight.grad);
        }
        Assert.Equal(before,Tensor.TotalCount);
    }

    [Fact]
    public void Invalid_factors_nonfinite_results_and_cancellation_leave_base_untouched()
    {
        NativeRuntimeBootstrap.Initialize();long before=Tensor.TotalCount;
        using(var scope=NewDisposeScope())
        {
            var weight=ones(new long[]{2,3});var up=ones(new long[]{2,1});var down=ones(new long[]{1,3});
            Assert.ThrowsAny<OperationCanceledException>(()=>LoraMath.Apply(weight,up,down,cancellationToken:new(true)));
            Assert.Throws<ArgumentException>(()=>LoraMath.Apply(weight,up,ones(new long[]{2,3})));
            Assert.Throws<ArgumentOutOfRangeException>(()=>LoraMath.Apply(weight,up,down,double.NaN));
            Assert.Throws<ArithmeticException>(()=>LoraMath.Apply(weight,full_like(up,float.NaN),down));
            Assert.Throws<ArgumentException>(()=>LoraMath.Apply(weight,up,down,mid:ones(new long[]{1,1})));
            Assert.All(weight.data<float>().ToArray(),v=>Assert.Equal(1,v));
        }
        Assert.Equal(before,Tensor.TotalCount);
    }

    private static Tensor? Optional(JsonElement value)=>value.ValueKind==JsonValueKind.Null?null:Read(value);
    private static Tensor Read(JsonElement value)=>tensor(value.GetProperty("values").EnumerateArray().Select(v=>v.GetSingle()).ToArray())
        .reshape(value.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()).ToArray());
    private static void Compare(Tensor actual,JsonElement expected)
    {
        Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(v=>v.GetInt64()),actual.shape);
        var values=expected.GetProperty("values").EnumerateArray().Select(v=>v.GetSingle()).ToArray();
        var data=actual.contiguous().data<float>().ToArray();Assert.Equal(values.Length,data.Length);
        for(int i=0;i<data.Length;i++)Assert.InRange(Math.Abs((double)data[i]-values[i]),0,1e-6+1e-6*Math.Abs(values[i]));
    }

    [Fact]
    public void Rank_scaling_is_applied_without_changing_borrowed_weights()
    {
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope();
        var weight = tensor(new[]{1f,2f,3f,4f}).reshape(2,2);
        var up = tensor(new[]{1f,2f}).reshape(2,1); var down = tensor(new[]{3f,4f}).reshape(1,2);
        using var result = LoraMath.Apply(weight,up,down,.5,2);
        Assert.Equal(new[]{4f,6f,9f,12f},result.data<float>().ToArray());
        result.fill_(0); Assert.Equal(new[]{1f,2f,3f,4f},weight.data<float>().ToArray());
    }

    [Fact]
    public void Adapter_gradients_survive_return_and_update_both_low_rank_factors()
    {
        NativeRuntimeBootstrap.Initialize(); long before=Tensor.TotalCount;
        using (var scope=NewDisposeScope())
        using (var grad=set_grad_enabled(true))
        {
            var weight=zeros(new long[]{2,2});
            var up=ones(new long[]{2,1},requires_grad:true); var down=ones(new long[]{1,2},requires_grad:true);
            using var result=LoraMath.Apply(weight,up,down,.5,2);
            var loss=result.square().mean(); loss.backward();
            Assert.All(up.grad!.data<float>().ToArray(),v=>Assert.Equal(1,v));
            Assert.All(down.grad!.data<float>().ToArray(),v=>Assert.Equal(1,v));
            Assert.Null(weight.grad); Assert.True(is_grad_enabled());
        }
        Assert.Equal(before,Tensor.TotalCount);
    }
}
