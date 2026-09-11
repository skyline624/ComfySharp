using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Reference-only process setting. Product startup does not set a global test thread policy.</summary>
internal static class SdReferenceRuntime
{
    private static readonly Lazy<bool> configured = new(() =>
    {
        NativeRuntimeBootstrap.Initialize();
        // libtorch permits configuring the inter-op pool only once per process.
        // The reference classes belong to the nonparallel native collection.
        set_num_interop_threads(1);
        return true;
    });

    internal static void Verify()
    {
        _ = configured.Value;
        Assert.Equal(1, get_num_interop_threads());
        Assert.Equal(1, get_num_threads());
    }
}
