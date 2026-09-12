using System.Text.Json.Nodes;
using ComfySharp.Contracts;
using ComfySharp.Core;
using ComfySharp.Inference;
using ComfySharp.Tokenization;
using static TorchSharp.torch;
using TorchTensor = TorchSharp.torch.Tensor;

namespace ComfySharp.Nodes.Tensor;

/// <summary>Stock SD1.5 Float32 workflow path with explicit CPU/CUDA placement. Other modes fail explicitly.</summary>
public static class Sd15Nodes
{
    // Frozen comfy/samplers.py order; retaining identifiers does not announce their execution support.
    private const string Samplers = "euler euler_cfg_pp euler_ancestral euler_ancestral_cfg_pp heun heunpp2 exp_heun_2_x0 exp_heun_2_x0_sde dpm_2 dpm_2_ancestral lms dpm_fast dpm_adaptive dpmpp_2s_ancestral dpmpp_2s_ancestral_cfg_pp dpmpp_sde dpmpp_sde_gpu dpmpp_2m dpmpp_2m_cfg_pp dpmpp_2m_sde dpmpp_2m_sde_gpu dpmpp_2m_sde_heun dpmpp_2m_sde_heun_gpu dpmpp_3m_sde dpmpp_3m_sde_gpu ddpm lcm ipndm ipndm_v deis cfgpp_ud10_ab res_multistep res_multistep_cfg_pp res_multistep_ancestral res_multistep_ancestral_cfg_pp gradient_estimation gradient_estimation_cfg_pp er_sde seeds_2 seeds_3 sa_solver sa_solver_pece ddim uni_pc uni_pc_bh2";
    private const string Schedulers = "simple sgm_uniform karras exponential ddim_uniform beta normal linear_quadratic kl_optimal";
    public static void Register(NodeRegistry registry, CheckpointFiles files, int cpuThreads = 16,
        long maxEstimatedPeakWeightBytes = 16L * 1024 * 1024 * 1024, string inferenceDevice = "cpu")
    {
        ArgumentNullException.ThrowIfNull(files);
        if (cpuThreads is < 1 or > 64) throw new ArgumentOutOfRangeException(nameof(cpuThreads));
        if (maxEstimatedPeakWeightBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxEstimatedPeakWeightBytes));
        if (inferenceDevice is not ("cpu" or "cuda:0")) throw new ArgumentException("Select inference device cpu or cuda:0.", nameof(inferenceDevice));
        foreach (var schema in Schemas(files.Names())) registry.Register(new Node(schema, files, cpuThreads, maxEstimatedPeakWeightBytes, inferenceDevice));
    }

    public static IReadOnlyList<NodeSchema> Schemas(IReadOnlyList<string> checkpointNames) =>
    [
        new("CheckpointLoaderSimple", "Load Checkpoint", "model/loaders", [Combo("ckpt_name", checkpointNames)],
            [new("MODEL"), new("CLIP"), new("VAE")], PythonModule: "nodes",
            Description: "Stock SD1.5 safetensors, EPS prediction, Float32 on the explicitly selected CPU or CUDA device. Other architectures remain unavailable."),
        new("CLIPTextEncode", "CLIP Text Encode (Prompt)", "model/conditioning",
            [new("text", "STRING", Options: new() { ["multiline"] = true, ["dynamicPrompts"] = true }), new("clip", "CLIP")],
            [new("CONDITIONING")], PythonModule: "nodes"),
        new("EmptyLatentImage", "Empty Latent Image", "model/latent",
            [Integer("width", 512, 16, 16384, 8), Integer("height", 512, 16, 16384, 8), Integer("batch_size", 1, 1, 4096)],
            [new("LATENT")], PythonModule: "nodes"),
        new("KSampler", "KSampler", "model/sampling",
            [new("model", "MODEL"), new("seed", "INT", Options: new() { ["default"] = 0, ["min"] = 0, ["max"] = ulong.MaxValue, ["control_after_generate"] = true }),
             Integer("steps", 20, 1, 10000), new("cfg", "FLOAT", Options: new() { ["default"] = 8.0, ["min"] = 0.0, ["max"] = 100.0, ["step"] = .1, ["round"] = .01 }),
             Combo("sampler_name", Samplers.Split(' ')), Combo("scheduler", Schedulers.Split(' ')), new("positive", "CONDITIONING"),
             new("negative", "CONDITIONING"), new("latent_image", "LATENT"),
             new("denoise", "FLOAT", Options: new() { ["default"] = 1.0, ["min"] = 0.0, ["max"] = 1.0, ["step"] = .01 })],
            [new("LATENT")], PythonModule: "nodes",
            Description: "Available execution: SD1.5 Float32 on CPU or CUDA, Euler/Heun without churn or DPM++ 2M, all nine listed schedulers, denoise in [0,1], one image up to 512x512, 1-100 requested steps. Expanded schedules are limited to 10000 steps; DDIM and beta may change the interval count. Nonfinite schedules or DPM++ 2M trajectories report an error."),
        new("VAEEncode", "VAE Encode", "model/latent", [new("pixels", "IMAGE"), new("vae", "VAE")], [new("LATENT")], PythonModule: "nodes"),
        new("VAEDecode", "VAE Decode", "model/latent", [new("samples", "LATENT"), new("vae", "VAE")], [new("IMAGE")], PythonModule: "nodes")
    ];

    private static InputSchema Combo(string name, IReadOnlyList<string> values) => new(name, "COMBO",
        Options: new() { ["options"] = new JsonArray(values.Select(v => (JsonNode?)JsonValue.Create(v)).ToArray()) });
    private static InputSchema Integer(string name, int value, int min, int max, int? step = null)
    {
        var options = new JsonObject { ["default"] = value, ["min"] = min, ["max"] = max };
        if (step is not null) options["step"] = step.Value;
        return new(name, "INT", Options: options);
    }

    private sealed class Node(NodeSchema schema, CheckpointFiles files, int threads, long budget, string inferenceDevice) : IRuntimeNode
    {
        public NodeSchema Schema => schema;
        public ValueTask<NodeExecutionOutput> ExecuteAsync(RuntimeNodeContext context,
            IReadOnlyDictionary<string, RuntimeValue> inputs, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string S(string key) => inputs[key].ToJson()!.GetValue<string>();
            int I(string key) => checked((int)PythonValues.Integer(inputs[key].ToJson()));
            double D(string key) => PythonValues.Float(inputs[key].ToJson());
            RuntimeValue Own(TorchTensor tensor)
            {
                var value = context.Own(tensor); tensor.DetachFromDisposeScope(); return value;
            }
            RuntimeValue[] outputs;
            if (schema.ClassType == "CheckpointLoaderSimple")
            {
                string path = files.Resolve(S("ckpt_name"));
                using var file = new SafeTensorFile(path);
                var plan = Sd15CheckpointLoader.Inspect(file, new() { UnclaimedTensors = Sd15UnclaimedTensorHandling.ReportAndIgnoreOutsideComponents }, cancellationToken);
                NativeRuntimeBootstrap.Initialize(); set_num_threads(threads);
                if (inferenceDevice == "cuda:0" && !cuda.is_available()) throw new NotSupportedException("CUDA inference was selected but is unavailable; no CPU fallback is performed.");
                var device = inferenceDevice == "cpu" ? CPU : new Device(TorchSharp.DeviceType.CUDA, 0);
                InferenceDevice.ConfigureFloat32(device);
                using var checkpoint = Sd15CheckpointLoader.Load(file, plan, budget, cancellationToken);
                using var sourceUnet = checkpoint.CreateUnet(); using var sourceClip = checkpoint.CreateClipEncoder(); using var sourceVae = checkpoint.CreateImageVae();
                outputs = [context.Own(sourceUnet.To(device, cancellationToken)), context.Own(sourceClip.To(device, cancellationToken)), context.Own(sourceVae.To(device, cancellationToken))];
                return ValueTask.FromResult(new NodeExecutionOutput(outputs, new JsonObject
                {
                    ["comfysharp_model"] = new JsonArray(new JsonObject { ["architecture"] = "SD1.5", ["backend"] = inferenceDevice, ["dtype"] = "Float32",
                        ["tf32_allowed"] = false,
                        ["component_devices"] = new JsonArray(outputs[0].GetNative<SdUnet>().Device.ToString(), outputs[1].GetNative<ComfyClipEncoder>().Device.ToString(), outputs[2].GetNative<ComfyImageVae>().Device.ToString()),
                        ["ignored_auxiliary_tensors"] = new JsonArray(plan.UnclaimedTensorNames.Select(n => (JsonNode?)JsonValue.Create(n)).ToArray()) })
                }));
            }
            NativeRuntimeBootstrap.Initialize();
            using var scope = NewDisposeScope();
            using var noGrad = no_grad();
            switch (schema.ClassType)
            {
                case "CLIPTextEncode":
                {
                    var clip = inputs["clip"].GetNative<ComfyClipEncoder>();
                    var tokenizer = new ComfyClipTokenizer(ClipTokenizer.CreateDefault(), clip.Profile);
                    using var encoded = clip.Encode(tokenizer.Tokenize(S("text"), cancellationToken: cancellationToken), cancellationToken: cancellationToken);
                    // CONDITIONING is the upstream nested list of [tensor, metadata], not an engine output-list.
                    outputs = [context.List([context.List([Own(encoded.Hidden.alias()),
                        context.Map(new Dictionary<string, RuntimeValue> { ["pooled_output"] = Own(encoded.Pooled.alias()) })])])];
                    break;
                }
                case "EmptyLatentImage":
                {
                    int width = I("width"), height = I("height"), batch = I("batch_size");
                    if (width is < 16 or > 16384 || height is < 16 or > 16384 || batch is < 1 or > 4096)
                        throw new ArgumentOutOfRangeException("LATENT dimensions are outside the upstream schema.");
                    // Explicit allocation guard before creating potentially multi-gigabyte tensors.
                    if (checked((long)batch * 4 * (height / 8) * (width / 8) * 4) > 256L * 1024 * 1024)
                        throw new NotSupportedException("EmptyLatentImage currently limits allocation to 256 MiB.");
                    outputs = [context.Map(new Dictionary<string, RuntimeValue>
                    {
                        ["samples"] = Own(zeros(new long[] { batch, 4, height / 8, width / 8 }, dtype: ScalarType.Float32, device: CPU)),
                        ["downscale_ratio_spacial"] = context.Json(JsonValue.Create(8))
                    })];
                    break;
                }
                case "KSampler":
                {
                    if (S("sampler_name") is not ("euler" or "heun" or "dpmpp_2m") || !SdScheduler.Names.Contains(S("scheduler"), StringComparer.Ordinal))
                        throw new NotSupportedException("KSampler currently executes Euler, Heun or DPM++ 2M with the ported SD schedulers.");
                    double denoise = D("denoise");
                    if (!double.IsFinite(denoise) || denoise is < 0 or > 1) throw new ArgumentOutOfRangeException("denoise");
                    int steps = I("steps"); double cfg = D("cfg");
                    if (steps is < 1 or > 100 || !double.IsFinite(cfg) || cfg is < 0 or > 100)
                        throw new NotSupportedException("KSampler currently requires 1-100 steps and CFG in [0,100].");
                    var latent = inputs["latent_image"].Properties;
                    var result = latent.Where(p => p.Key is not ("downscale_ratio_spacial" or "downscale_ratio_temporal"))
                        .ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal);
                    // Frozen CFGGuider.sample returns the original raw latent for an empty schedule.
                    if (denoise == 0)
                    {
                        result["samples"] = Own(latent["samples"].GetNative<TorchTensor>().alias());
                        outputs = [context.Map(result)]; break;
                    }
                    if (latent.ContainsKey("noise_mask") || latent.ContainsKey("batch_index"))
                        throw new NotSupportedException("Masked sampling and batch-index noise are not yet ported.");
                    if (latent.TryGetValue("downscale_ratio_spacial", out var ratio) && PythonValues.Integer(ratio.ToJson()) != 8 ||
                        latent.ContainsKey("downscale_ratio_temporal"))
                        throw new NotSupportedException("Only SD1.5 spatial latents with downscale ratio 8 are supported.");
                    var model = inputs["model"].GetNative<SdUnet>();
                    using var raw = latent["samples"].GetNative<TorchTensor>().to(model.Device, copy: true); ValidateShape(raw);
                    ulong seed = ulong.Parse(inputs["seed"].ToJson()!.ToJsonString(), System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture);
                    using var cpuNoise = NativeMath.CpuNoise(raw.shape, seed, cancellationToken);
                    using var noise = cpuNoise.to(model.Device, copy: true);
                    using var scaled = SdSamplingMath.ProcessLatentIn(raw, SdSamplingMath.Sd15LatentScale, cancellationToken);
                    var sampling = SdDiscreteSampling.Default;
                    using var cpuSigmas = SdScheduler.Create(S("scheduler"), steps, denoise, sampling, cancellationToken);
                    using var sigmas = cpuSigmas.to(model.Device, copy: true);
                    using var first = sigmas[0];
                    using var initial = SdSamplingMath.NoiseScaling(noise, scaled, first,
                        SdKarrasSchedule.UsesMaximumNoise(first.item<float>(), sampling.SigmaMax), cancellationToken);
                    using var denoiser = new SdDenoiser(model, SdPredictionKind.Epsilon, sampling);
                    var guidance = new SdGuidanceOptions { Scale = cfg, BatchMode = SdGuidanceBatchMode.Separate };
                    TorchTensor ExecuteSampler()
                    {
                        if (S("sampler_name") == "dpmpp_2m")
                        {
                            using var sampler = new SdDpmpp2MSampler(denoiser);
                            return sampler.Sample(initial, sigmas, Conditioning(inputs["positive"]), Conditioning(inputs["negative"]), guidance, cancellationToken);
                        }
                        if (S("sampler_name") == "heun")
                        {
                            using var sampler = new SdHeunSampler(denoiser);
                            return sampler.Sample(initial, sigmas, Conditioning(inputs["positive"]), Conditioning(inputs["negative"]), guidance, cancellationToken);
                        }
                        using var euler = new SdEulerSampler(denoiser);
                        return euler.Sample(initial, sigmas, Conditioning(inputs["positive"]), Conditioning(inputs["negative"]), guidance, cancellationToken);
                    }
                    using var sampled = ExecuteSampler();
                    result["samples"] = Own(SdSamplingMath.ProcessLatentOut(sampled, SdSamplingMath.Sd15LatentScale, cancellationToken));
                    outputs = [context.Map(result)];
                    break;
                }
                case "VAEEncode":
                {
                    var vae = inputs["vae"].GetNative<ComfyImageVae>();
                    var source = inputs["pixels"].GetNative<TorchTensor>();
                    var shape = source.shape;
                    if (shape.Length != 4 || shape[0] != 1 || shape[1] is < 32 or > 519 || shape[2] is < 32 or > 519)
                        throw new NotSupportedException("SD1.5 VAEEncode currently requires one image, cropped to 32-512 pixels per dimension.");
                    using var pixels = source.to(vae.Device, copy: true);
                    outputs = [context.Map(new Dictionary<string, RuntimeValue> { ["samples"] = Own(vae.Encode(pixels, cancellationToken)) })];
                    break;
                }
                case "VAEDecode":
                {
                    var vae = inputs["vae"].GetNative<ComfyImageVae>();
                    using var raw = inputs["samples"].Properties["samples"].GetNative<TorchTensor>().to(vae.Device, copy: true); ValidateShape(raw);
                    using var decoded = vae.Decode(raw, cancellationToken);
                    // IMAGE/PNG and media nodes currently consume CPU buffers; inference remains on the selected device.
                    outputs = [Own(decoded.to(CPU, copy: true))];
                    break;
                }
                default: throw new InvalidOperationException("Unregistered SD1.5 operation.");
            }
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult(new NodeExecutionOutput(outputs));
        }
        private static void ValidateShape(TorchTensor raw)
        {
            long[] shape = raw.shape;
            if (shape.Length != 4 || shape[0] != 1 || shape[1] != 4 || shape[2] is < 4 or > 64 || shape[3] is < 4 or > 64)
                throw new NotSupportedException("SD1.5 execution currently requires one image, 32-512 pixels per dimension.");
        }
        private static TorchTensor Conditioning(RuntimeValue value)
        {
            if (value.Kind != RuntimeValueKind.List || value.Items.Count != 1 || value.Items[0].Kind != RuntimeValueKind.List || value.Items[0].Items.Count != 2)
                throw new NotSupportedException("KSampler currently supports one global conditioning entry per prompt.");
            var entry = value.Items[0].Items;
            if (entry[1].Kind != RuntimeValueKind.Map || entry[1].Properties.Keys.Any(k => k != "pooled_output"))
                throw new NotSupportedException("Regional, masked and scheduled conditioning are not yet ported.");
            return entry[0].GetNative<TorchTensor>();
        }
    }
}
