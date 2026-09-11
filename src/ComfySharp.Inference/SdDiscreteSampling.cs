using TorchSharp;
using static TorchSharp.torch;

namespace ComfySharp.Inference;

/// <summary>The default 1,000-step SD1/SD2 discrete schedule from frozen ModelSamplingDiscrete.
/// This profile has linear square-root betas, sigma_data=1 and no zero-terminal-SNR rescaling.</summary>
public sealed class SdDiscreteSampling
{
    private sealed record Tables(float[] Sigmas, float[] LogSigmas);
    private static readonly Lazy<Tables> tables = new(CreateTables);
    public static SdDiscreteSampling Default { get; } = new();
    public const int Count = 1000;
    public IReadOnlyList<float> Sigmas => Array.AsReadOnly(tables.Value.Sigmas);
    public IReadOnlyList<float> LogSigmas => Array.AsReadOnly(tables.Value.LogSigmas);
    public float SigmaMin => tables.Value.Sigmas[0];
    public float SigmaMax => tables.Value.Sigmas[^1];

    private static Tables CreateTables()
    {
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        var betas = linspace(Math.Sqrt(0.00085), Math.Sqrt(0.012), Count,
            dtype: ScalarType.Float64, device: CPU).pow(2);
        var cumulativeAlpha = (1.0 - betas).cumprod(0);
        var sigmas = ((1.0 - cumulativeAlpha) / cumulativeAlpha).pow(0.5);
        // Source logs the F64 schedule BEFORE converting each table to F32.
        return new(sigmas.to_type(ScalarType.Float32).data<float>().ToArray(),
            sigmas.log().to_type(ScalarType.Float32).data<float>().ToArray());
    }

    /// <summary>Nearest log-sigma index, with source first-index tie breaking. Zero maps to index zero.
    /// Input is a nonnegative finite CPU/F32 scalar or vector; output is an independently owned Int64 tensor.</summary>
    public Tensor Timestep(Tensor sigma, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        SdSamplingMath.ValidateSigma(sigma);
        var logTable = tensor(tables.Value.LogSigmas, dtype: ScalarType.Float32, device: CPU).unsqueeze(1);
        var distances = sigma.log().reshape(-1) - logTable;
        var result = distances.abs().argmin(0).reshape(sigma.shape);
        cancellationToken.ThrowIfCancellationRequested();
        return result.DetachFromDisposeScope();
    }

    /// <summary>Clamped interpolation in the source F32 log table. Fractional times are supported.
    /// Accepts CPU F32/F64/Int32/Int64 scalars or vectors; infinities clamp to the endpoint, NaN is rejected.</summary>
    public Tensor Sigma(Tensor timestep, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        NativeRuntimeBootstrap.Initialize();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        ArgumentNullException.ThrowIfNull(timestep);
        if (timestep.device_type != DeviceType.CPU || timestep.is_sparse || timestep.dim() > 1 ||
            timestep.dtype is not (ScalarType.Float32 or ScalarType.Float64 or ScalarType.Int32 or ScalarType.Int64))
            throw new ArgumentException("Discrete timesteps require a dense CPU numeric scalar or vector.", nameof(timestep));
        var time = timestep.to_type(ScalarType.Float32).clamp(0, Count - 1).reshape(-1);
        if (time.isnan().any().item<bool>()) throw new ArgumentException("Timesteps cannot contain NaN.", nameof(timestep));
        var low = time.floor().to_type(ScalarType.Int64);
        var high = time.ceil().to_type(ScalarType.Int64);
        var weight = time.frac();
        var logTable = tensor(tables.Value.LogSigmas, dtype: ScalarType.Float32, device: CPU);
        var logSigma = (1.0 - weight) * logTable.index_select(0, low) + weight * logTable.index_select(0, high);
        var result = logSigma.exp().reshape(timestep.shape);
        cancellationToken.ThrowIfCancellationRequested();
        return result.DetachFromDisposeScope();
    }

    public double PercentToSigma(double percent, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (double.IsNaN(percent)) throw new ArgumentOutOfRangeException(nameof(percent));
        if (percent <= 0) return 999999999.9;
        if (percent >= 1) return 0;
        NativeRuntimeBootstrap.Initialize();
        using var time = tensor((float)((1.0 - percent) * 999.0), dtype: ScalarType.Float32, device: CPU);
        using var result = Sigma(time, cancellationToken);
        return result.item<float>();
    }
}
