using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Opt-in managed copies of CPU training evidence. Never computes expected values.</summary>
internal sealed class SdTrainingTrace : IDisposable
{
    private readonly string path;
    private readonly Dictionary<string,object> records=new(StringComparer.Ordinal);
    private string phase="initial";
    private SdTrainingTrace(string path)=>this.path=path;
    internal static SdTrainingTrace? Open(int caseIndex)
    {
        string? root=Environment.GetEnvironmentVariable("COMFYSHARP_TRAINING_TRACE_DIR");
        if(string.IsNullOrWhiteSpace(root))return null;
        if(Environment.GetEnvironmentVariable("COMFYSHARP_TRAINING_INTEROP_ONE")=="1")SdReferenceRuntime.Verify();
        Directory.CreateDirectory(root);
        string path=Path.Combine(root,$"case-{caseIndex}.json");
        if(File.Exists(path))throw new IOException("Training trace destination already exists.");
        return new(path);
    }
    internal void Phase(string value)=>phase=value;
    internal void Capture(string name,Tensor value)=>records.Add(phase+"/"+name,Snapshot(value));
    internal void Weights(UnetWeightSet bank)
    {
        records.Add("baseWeights",UnetWeightSchema.Describe(bank.Config).Keys.ToDictionary(n=>n,n=>
        {
            var t=bank.GetTensor(n);
            return new{shape=t.shape,stride=t.stride(),aligned64=CpuModelWeightBank.IsAligned(t),sha256=Hash(t)};
        }));
    }
    private static string Hash(Tensor t)=>Convert.ToHexStringLower(SHA256.HashData(t.bytes));
    private static object Snapshot(Tensor t)
    {
        // All observed boundaries and leaf gradients are dense CPU F32; no clone/reshape/native allocation.
        if(t.device_type!=DeviceType.CPU||t.dtype!=ScalarType.Float32||!t.is_contiguous())throw new InvalidDataException("Unexpected training trace tensor layout.");
        return new{shape=t.shape,stride=t.stride(),sha256=Hash(t),values=t.data<float>().ToArray()};
    }
    public void Dispose()
    {
        using var stream=new FileStream(path,FileMode.CreateNew,FileAccess.Write,FileShare.None);
        JsonSerializer.Serialize(stream,new{target=SdSamplingReferenceTests.Target,processArchitecture=RuntimeInformation.ProcessArchitecture.ToString(),
            threads=get_num_threads(),interopThreads=get_num_interop_threads(),avx2=System.Runtime.Intrinsics.X86.Avx2.IsSupported,
            avx512=System.Runtime.Intrinsics.X86.Avx512F.IsSupported,atenCpuCapability=Environment.GetEnvironmentVariable("ATEN_CPU_CAPABILITY"),records});
    }
}
