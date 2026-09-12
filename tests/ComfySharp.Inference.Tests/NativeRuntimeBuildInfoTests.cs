using System.Runtime.InteropServices;
using ComfySharp.Inference;
using Xunit;

namespace ComfySharp.Inference.Tests;

public sealed class NativeRuntimeBuildInfoTests
{
    [Fact]
    public void Strings_are_copied_before_native_storage_is_reused()
    {
        nint capability=Marshal.StringToCoTaskMemUTF8("AVX2"), config=Marshal.StringToCoTaskMemUTF8("libtorch build — CPU");
        NativeRuntimeBuildInfo value;
        try
        {
            value=NativeRuntimeBuildInfo.ReadCore(()=>210000,(out nint c,out nint b)=>{c=capability;b=config;return 0;});
        }
        finally{Marshal.FreeCoTaskMem(capability);Marshal.FreeCoTaskMem(config);}
        Assert.Equal("AVX2",value.CpuCapability);Assert.Equal("libtorch build — CPU",value.BuildConfiguration);
    }

    [Theory]
    [InlineData(0)] [InlineData(209000)]
    public void Wrong_abi_is_rejected_before_reading_native_strings(int abi)
    {
        Assert.Throws<NotSupportedException>(()=>NativeRuntimeBuildInfo.ReadCore(()=>abi,
            (out nint c,out nint b)=>throw new Exception("Read must not run.")));
    }

    [Fact]
    public void Native_error_is_copied_and_reported_without_dereferencing_outputs()
    {
        nint error=Marshal.StringToCoTaskMemUTF8("configuration unavailable");
        try
        {
            var failure=Assert.Throws<InvalidOperationException>(()=>NativeRuntimeBuildInfo.ReadCore(()=>210000,
                (out nint c,out nint b)=>{c=0;b=0;return error;}));
            Assert.Contains("configuration unavailable",failure.Message);
        }
        finally{Marshal.FreeCoTaskMem(error);}
    }

    [Fact]
    public void Successful_status_cannot_hide_missing_native_identity()
    {
        Assert.Throws<InvalidDataException>(()=>NativeRuntimeBuildInfo.ReadCore(()=>210000,
            (out nint c,out nint b)=>{c=0;b=0;return 0;}));
    }
}
