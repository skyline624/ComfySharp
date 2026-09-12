using ComfySharp.RuntimeProbe;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

/// <summary>Ownership and validation controls. Synthetic weights do not qualify model inference.</summary>
[Collection("Classical VAE")]
public sealed class ModelDeviceTests
{
    [Fact]
    public void Float32_cuda_policy_disables_tf32_and_cpu_leaves_process_flags_unchanged()
    {
        // This tests libtorch policy flags using the CPU bundle, not CUDA kernel execution.
        NativeRuntimeBootstrap.Initialize();
        bool matmul = backends.cuda.matmul.allow_tf32, convolution = backends.cudnn.allow_tf32;
        try
        {
            backends.cuda.matmul.allow_tf32 = true; backends.cudnn.allow_tf32 = true;
            InferenceDevice.ConfigureFloat32(CPU);
            Assert.True(backends.cuda.matmul.allow_tf32); Assert.True(backends.cudnn.allow_tf32);
            InferenceDevice.ConfigureFloat32(new Device(DeviceType.CUDA, 0));
            Assert.False(backends.cuda.matmul.allow_tf32); Assert.False(backends.cudnn.allow_tf32);
        }
        finally { backends.cuda.matmul.allow_tf32 = matmul; backends.cudnn.allow_tf32 = convolution; }
    }

    [Fact]
    public void Same_device_graph_owners_survive_the_original_banks_and_graphs()
    {
        NativeRuntimeBootstrap.Initialize(); int previous = get_num_threads(); set_num_threads(1);
        long before = Tensor.TotalCount;
        try
        {
            var clipConfig = Sd15PipelineDiagnostic.ClipConfig;
            var unetConfig = Sd15PipelineExecution.UnetConfig;
            var vaeConfig = Sd15PipelineExecution.VaeConfig;
            var clipPlan = SdSyntheticWeightBuilder.Describe(ClipWeightSchema.Describe(clipConfig));
            var unetPlan = SdSyntheticWeightBuilder.DescribeUnet(unetConfig);
            var vaePlan = SdSyntheticWeightBuilder.Describe(ClassicalVaeWeightSchema.Describe(vaeConfig));
            using (var scope = NewDisposeScope())
            {
                var clipBank = SdSyntheticWeightBuilder.Create(clipPlan, new(clipPlan.ResidentBytes), v => ClipWeightSet.FromOwnedTensors(clipConfig, v));
                var unetBank = SdSyntheticWeightBuilder.Create(unetPlan, new(unetPlan.ResidentBytes), v => UnetWeightSet.FromOwnedTensors(unetConfig, v));
                var vaeBank = SdSyntheticWeightBuilder.Create(vaePlan, new(vaePlan.ResidentBytes), v => ClassicalVaeWeightSet.FromOwnedTensors(vaeConfig, v));
                using var originalClip = new ClipTextEncoder(clipBank.Weights); using var originalUnet = new SdUnet(unetBank.Weights); using var originalVae = new ClassicalVae(vaeBank.Weights);
                clipBank.Dispose(); unetBank.Dispose(); vaeBank.Dispose();
                using var clip = originalClip.To(CPU); using var unet = originalUnet.To(CPU); using var vae = originalVae.To(CPU);
                originalClip.Dispose(); originalUnet.Dispose(); originalVae.Dispose();
                Assert.True(InferenceDevice.Same(CPU, clip.Device)); Assert.True(InferenceDevice.Same(CPU, unet.Device)); Assert.True(InferenceDevice.Same(CPU, vae.Device));
                using var encoded = clip.Forward([Enumerable.Repeat(0, ClipTextConfig.MaxPositions).ToArray()]);
                using var latent = zeros(new long[] { 1, 4, 8, 8 }); using var time = zeros(new long[] { 1 });
                using var prediction = unet.Forward(latent, time, encoded.FinalHidden);
                using var image = vae.Decode(latent);
                Assert.Equal(new long[] { 1, 4, 8, 8 }, prediction.shape); Assert.True(prediction.isfinite().all().item<bool>());
                Assert.Equal(new long[] { 1, 3, 64, 64 }, image.shape); Assert.True(image.isfinite().all().item<bool>());
                using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
                Assert.ThrowsAny<OperationCanceledException>(() => clip.To(CPU, cancelled.Token));
                Assert.ThrowsAny<OperationCanceledException>(() => unet.To(CPU, cancelled.Token));
                Assert.ThrowsAny<OperationCanceledException>(() => vae.To(CPU, cancelled.Token));
                var unsupported = new Device(DeviceType.MPS);
                Assert.Throws<NotSupportedException>(() => clip.To(unsupported));
                Assert.Throws<NotSupportedException>(() => unet.To(unsupported));
                Assert.Throws<NotSupportedException>(() => vae.To(unsupported));
            }
            Assert.Equal(before, Tensor.TotalCount);
        }
        finally { set_num_threads(previous); }
    }

    [Fact]
    public void Device_identity_distinguishes_cuda_indices_and_canonicalizes_default_cuda()
    {
        Assert.True(InferenceDevice.Same(CPU, new Device(DeviceType.CPU, 0)));
        Assert.False(InferenceDevice.Same(CPU, new Device(DeviceType.CUDA, 0)));
        Assert.False(InferenceDevice.Same(new Device(DeviceType.CUDA, 0), new Device(DeviceType.CUDA, 1)));
        Assert.Equal(0, InferenceDevice.Validate(new Device(DeviceType.CUDA)).index);
    }
}
