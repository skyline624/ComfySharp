using static TorchSharp.torch;
namespace ComfySharp.Inference;

public static class SdScheduler
{
    public static IReadOnlyList<string> Names { get; } = Array.AsReadOnly(new[]
        { "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal" });

    /// <summary>Frozen KSampler schedules for default SD discrete sampling, including variable-length
    /// DDIM/beta results and set_steps partial-denoise slicing. Result ownership follows the caller's scope.</summary>
    public static Tensor Create(string scheduler, int steps, double denoise, SdDiscreteSampling sampling,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested(); ArgumentNullException.ThrowIfNull(sampling);
        if (!Names.Contains(scheduler, StringComparer.Ordinal)) throw new NotSupportedException("Unknown SD scheduler: " + scheduler);
        if (steps is < 1 or > 10000) throw new ArgumentOutOfRangeException(nameof(steps));
        if (!double.IsFinite(denoise) || denoise is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(denoise));
        NativeRuntimeBootstrap.Initialize(); using var scope = NewDisposeScope();
        if (denoise == 0) return empty(new long[] { 0 }, dtype: ScalarType.Float32).MoveToOuterDisposeScope();
        double expanded = denoise > .9999 ? steps : Math.Truncate(steps / denoise);
        if (expanded > 10000) throw new NotSupportedException("Expanded SD schedules currently support at most 10000 steps.");
        int n = (int)expanded;
        Tensor full = scheduler switch
        {
            "karras" => SigmaSchedules.Karras(n, sampling.SigmaMin, sampling.SigmaMax, cancellationToken: cancellationToken),
            "exponential" => SigmaSchedules.Exponential(n, sampling.SigmaMin, sampling.SigmaMax, cancellationToken),
            "normal" or "sgm_uniform" => Normal(n, scheduler == "sgm_uniform", sampling, cancellationToken),
            "simple" or "ddim_uniform" or "beta" => FromTable(scheduler, n, sampling, cancellationToken),
            "linear_quadratic" => LinearQuadratic(n, sampling),
            "kl_optimal" => KlOptimal(n, sampling),
            _ => throw new NotSupportedException(scheduler)
        };
        cancellationToken.ThrowIfCancellationRequested();
        if (!full.isfinite().all().item<bool>()) throw new ArithmeticException($"{scheduler} produces a nonfinite schedule for {n} steps in the frozen source.");
        // The full source DDIM schedule can be longer than steps+1. Only partial denoise slices it.
        if (denoise > .9999) return full.MoveToOuterDisposeScope();
        long count = Math.Min(full.shape[0], steps + 1L);
        return full.narrow(0, full.shape[0] - count, count).clone().MoveToOuterDisposeScope();
    }

    private static Tensor Normal(int n, bool sgm, SdDiscreteSampling sampling, CancellationToken token)
    {
        using var scope = NewDisposeScope();
        // This type represents the default SD table: sigma_max/min map to times 999/0.
        var times = linspace(SdDiscreteSampling.Count - 1, 0, sgm ? n + 1 : n, dtype: ScalarType.Float32);
        var values = new float[n + 1];
        for (int i = 0; i < n; i++)
        {
            token.ThrowIfCancellationRequested();
            using var time = times[i]; using var sigma = sampling.Sigma(time, token);
            values[i] = sigma.item<float>();
        }
        return tensor(values).MoveToOuterDisposeScope();
    }

    private static Tensor FromTable(string name, int n, SdDiscreteSampling sampling, CancellationToken token)
    {
        var table = sampling.Sigmas; var values = new List<float>(Math.Min(n + 1, 1001));
        if (name == "ddim_uniform")
        {
            values.Add(0);
            for (int index = 1; index < table.Count; index += Math.Max(table.Count / n, 1))
            { token.ThrowIfCancellationRequested(); values.Add(table[index]); }
            values.Reverse();
        }
        else
        {
            int previous = -1; double stride = (double)table.Count / n, probabilityStep = 1.0 / n;
            for (int i = 0; i < n; i++)
            {
                token.ThrowIfCancellationRequested();
                int index = name == "simple" ? table.Count - 1 - (int)(i * stride)
                    : (int)Math.Round(BetaQuantile(1 - i * probabilityStep) * (table.Count - 1), MidpointRounding.ToEven);
                if (name != "beta" || index != previous) values.Add(table[index]);
                previous = index;
            }
            values.Add(0);
        }
        return tensor(values.ToArray());
    }

    // For KSampler's fixed Beta(.6,.6), integrate t^(a-1)*(1-t)^(a-1) using
    // the binomial series on [0,.5]. Symmetry normalizes F(.5)=.5 without a gamma library.
    // Bisection is bounded and all arithmetic stays double until the final nearest-even table index.
    private static double BetaIntegral(double x)
    {
        const double a = .6; double term = 1, sum = 1 / a;
        for (int k = 1; k <= 128; k++)
        {
            term *= (k - a) * x / k;
            double next = sum + term / (a + k);
            if (next == sum) break;
            sum = next;
        }
        return Math.Pow(x, a) * sum;
    }
    private static readonly double betaHalfIntegral = BetaIntegral(.5);
    private static double BetaQuantile(double p)
    {
        if (p <= 0) return 0;
        if (p >= 1) return 1;
        // The exact median must stay .5: the next nearest-even index rounds 499.5 to 500.
        if (p == .5) return .5;
        bool mirror = p > .5; double target = mirror ? 1 - p : p, low = 0, high = .5;
        for (int i = 0; i < 64; i++)
        {
            double middle = (low + high) / 2;
            double cdf = .5 * BetaIntegral(middle) / betaHalfIntegral;
            if (cdf < target) low = middle; else high = middle;
        }
        double answer = (low + high) / 2; return mirror ? 1 - answer : answer;
    }

    private static Tensor LinearQuadratic(int n, SdDiscreteSampling sampling)
    {
        using var scope = NewDisposeScope(); var values = new float[n + 1];
        if (n == 1) values[0] = 1;
        else
        {
            int linear = n / 2, quadratic = n - linear; const double threshold = .025;
            double difference = linear - threshold * n;
            double quadraticCoefficient = difference / (linear * (double)quadratic * quadratic);
            double linearCoefficient = threshold / linear - 2 * difference / (quadratic * (double)quadratic);
            double constant = quadraticCoefficient * (linear * linear);
            for (int i = 0; i < n; i++)
                values[i] = (float)(1 - (i < linear ? i * threshold / linear
                    : quadraticCoefficient * (i * i) + linearCoefficient * i + constant));
        }
        return (tensor(values) * sampling.SigmaMax).MoveToOuterDisposeScope();
    }

    private static Tensor KlOptimal(int n, SdDiscreteSampling sampling)
    {
        using var scope = NewDisposeScope();
        var positions = arange(n, dtype: ScalarType.Float32) / (n - 1);
        var sigmas = (positions * Math.Atan(sampling.SigmaMin) + (1 - positions) * Math.Atan(sampling.SigmaMax)).tan();
        return cat(new[] { sigmas, zeros(new long[] { 1 }, dtype: ScalarType.Float32) }).MoveToOuterDisposeScope();
    }
}
