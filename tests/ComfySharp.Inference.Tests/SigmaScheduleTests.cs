using ComfySharp.Inference;
using Xunit;
using static TorchSharp.torch;

namespace ComfySharp.Inference.Tests;

public sealed class SigmaScheduleTests
{
    [Fact]
    public void KarrasUnitRhoHasExactLinearSpacingAndTerminalZero()
    {
        using var result = SigmaSchedules.Karras(5, 1, 9, rho: 1);
        Assert.Equal(new float[] { 9, 7, 5, 3, 1, 0 }, result.data<float>().ToArray());
    }

    [Fact]
    public void KarrasSupportsZeroAndReversedBoundsWithoutReordering()
    {
        using var zeroMin = SigmaSchedules.Karras(3, 0, 4, rho: 1);
        using var ascending = SigmaSchedules.Karras(3, 4, 0, rho: 1);
        Assert.Equal(new float[] { 4, 2, 0, 0 }, zeroMin.data<float>().ToArray());
        Assert.Equal(new float[] { 0, 2, 4, 0 }, ascending.data<float>().ToArray());
    }

    [Fact]
    public void ExponentialConstantBoundsProducePlateauFollowedByZero()
    {
        using var result = SigmaSchedules.Exponential(4, 1, 1);
        Assert.Equal(new float[] { 1, 1, 1, 1, 0 }, result.data<float>().ToArray());
    }

    [Fact]
    public void ExponentialMidpointIsGeometricMean()
    {
        using var result = SigmaSchedules.Exponential(3, 1, 16);
        AssertClose([16, 4, 1, 0], result.data<float>().ToArray());
    }

    [Fact]
    public void PolyexponentialZeroRhoRetainsMaximumThroughLastNonterminalPoint()
    {
        using var result = SigmaSchedules.Polyexponential(5, 1, 16, rho: 0);
        AssertClose([16, 16, 16, 16, 16, 0], result.data<float>().ToArray());
    }

    [Fact]
    public void PolyexponentialSquaredRampHasIndependentKnownValues()
    {
        using var result = SigmaSchedules.Polyexponential(3, 1, 16, rho: 2);
        AssertClose([16, 2, 1, 0], result.data<float>().ToArray());
    }

    [Fact]
    public void LaplaceZeroBetaProducesPlateauWithoutAddedZero()
    {
        using var result = SigmaSchedules.Laplace(5, 0.25, 4, mu: 0, beta: 0);
        Assert.Equal(new float[] { 1, 1, 1, 1, 1 }, result.data<float>().ToArray());
    }

    [Fact]
    public void LaplaceClampsExtremesAndPreservesCentralValue()
    {
        using var result = SigmaSchedules.Laplace(3, 0.25, 4);
        Assert.Equal(new float[] { 4, 1, 0.25f }, result.data<float>().ToArray());
    }

    [Fact]
    public void LaplaceReversedBoundsRetainUpstreamClampBehavior()
    {
        using var result = SigmaSchedules.Laplace(3, 4, 0.25);
        Assert.Equal(new float[] { 0.25f, 0.25f, 0.25f }, result.data<float>().ToArray());
    }

    [Fact]
    public void VpZeroCoefficientsAllowMultipleZeros()
    {
        using var result = SigmaSchedules.VP(3, betaD: 0, betaMin: 0);
        Assert.Equal(new float[] { 0, 0, 0, 0 }, result.data<float>().ToArray());
    }

    [Fact]
    public void VpAtZeroTerminalTimeKeepsBothComputedAndAppendedZeros()
    {
        using var result = SigmaSchedules.VP(2, betaD: 0, betaMin: Math.Log(2), epsS: 0);
        AssertClose([1, 0, 0], result.data<float>().ToArray());
    }

    [Fact]
    public void VpSmallCoefficientAvoidsCancellationInExpMinusOne()
    {
        using var result = SigmaSchedules.VP(1, betaD: 0, betaMin: 1e-12, epsS: 0);
        Assert.InRange(result.data<float>()[0], 0.99999e-6f, 1.00001e-6f);
        Assert.Equal(0, result.data<float>()[1]);
    }

    [Theory]
    [InlineData("karras")]
    [InlineData("exponential")]
    [InlineData("polyexponential")]
    [InlineData("laplace")]
    [InlineData("vp")]
    public void SingleStepReturnsFirstPointAndOnlyDocumentedTerminalZero(string name)
    {
        using var result = Generate(name, 1);
        Assert.Equal(name == "laplace" ? 1 : 2, result.numel());
        Assert.True(result.data<float>()[0] > 0);
        if (name != "laplace") Assert.Equal(0, result.data<float>()[1]);
    }

    [Theory]
    [InlineData("karras")]
    [InlineData("exponential")]
    [InlineData("polyexponential")]
    [InlineData("laplace")]
    [InlineData("vp")]
    public void OutputHasCpuFloat32ContractAndLivesInCallersDisposeScope(string name)
    {
        NativeRuntimeBootstrap.Initialize();
        Tensor result;
        using (var callerScope = NewDisposeScope())
        {
            result = Generate(name, 20);
            Assert.Equal(ScalarType.Float32, result.dtype);
            Assert.Equal("cpu", result.device.ToString().ToLowerInvariant());
            Assert.Equal(new long[] { name == "laplace" ? 20 : 21 }, result.shape);
            Assert.False(result.IsInvalid);
            Assert.All(result.data<float>().ToArray(), value => Assert.True(float.IsFinite(value)));
        }
        Assert.True(result.IsInvalid);
    }

    [Theory]
    [InlineData("karras")]
    [InlineData("exponential")]
    [InlineData("polyexponential")]
    [InlineData("laplace")]
    [InlineData("vp")]
    public void SeparateCallsOwnSeparateWrappersAndStorage(string name)
    {
        using var first = Generate(name, 3);
        using var second = Generate(name, 3);
        var expected = second.data<float>().ToArray();
        first.data<float>()[0] = -123;
        Assert.Equal(expected, second.data<float>().ToArray());
        first.Dispose();
        Assert.True(first.IsInvalid);
        Assert.False(second.IsInvalid);
        Assert.Equal(expected, second.data<float>().ToArray());
    }

    [Theory]
    [InlineData("karras")]
    [InlineData("exponential")]
    [InlineData("polyexponential")]
    [InlineData("laplace")]
    [InlineData("vp")]
    public void StepLimitsAndPreCancellationAreExplicit(string name)
    {
        Assert.Equal("steps", Assert.Throws<ArgumentOutOfRangeException>(() => Generate(name, 0)).ParamName);
        Assert.Equal("steps", Assert.Throws<ArgumentOutOfRangeException>(() => Generate(name, -1)).ParamName);
        Assert.Equal("steps", Assert.Throws<ArgumentOutOfRangeException>(() => Generate(name, 10001)).ParamName);
        Assert.Throws<OperationCanceledException>(() => Generate(name, 20, new CancellationToken(true)));
        using var maximum = Generate(name, 10000);
        Assert.Equal(name == "laplace" ? 10000 : 10001, maximum.numel());
    }

    [Fact]
    public void InvalidMathematicalDomainsAreRejectedWithParameterNames()
    {
        Assert.Equal("rho", Assert.Throws<ArgumentOutOfRangeException>(() => SigmaSchedules.Karras(3, 1, 4, 0)).ParamName);
        Assert.Equal("sigmaMin", Assert.Throws<ArgumentOutOfRangeException>(() => SigmaSchedules.Karras(3, -1, 4)).ParamName);
        Assert.Equal("sigmaMin", Assert.Throws<ArgumentOutOfRangeException>(() => SigmaSchedules.Exponential(3, 0, 4)).ParamName);
        Assert.Equal("sigmaMax", Assert.Throws<ArgumentOutOfRangeException>(() => SigmaSchedules.Polyexponential(3, 1, 0)).ParamName);
        Assert.Equal("rho", Assert.Throws<ArgumentOutOfRangeException>(() => SigmaSchedules.Polyexponential(3, 1, 4, -1)).ParamName);
        Assert.Equal("beta", Assert.Throws<ArgumentOutOfRangeException>(() => SigmaSchedules.Laplace(3, 1, 4, beta: -1)).ParamName);
        Assert.Equal("betaD", Assert.Throws<ArgumentOutOfRangeException>(() => SigmaSchedules.VP(3, betaD: -1)).ParamName);
        Assert.Equal("betaMin", Assert.Throws<ArgumentOutOfRangeException>(() => SigmaSchedules.VP(3, betaMin: -1)).ParamName);
        Assert.Equal("epsS", Assert.Throws<ArgumentOutOfRangeException>(() => SigmaSchedules.VP(3, epsS: 1.1)).ParamName);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void NonFiniteInputsAreNotSilentlyClampedOrConverted(double value)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => SigmaSchedules.Karras(3, 1, value));
        Assert.Throws<ArgumentOutOfRangeException>(() => SigmaSchedules.Exponential(3, value, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => SigmaSchedules.Polyexponential(3, 1, 4, value));
        Assert.Throws<ArgumentOutOfRangeException>(() => SigmaSchedules.Laplace(3, 1, 4, mu: value));
        Assert.Throws<ArgumentOutOfRangeException>(() => SigmaSchedules.VP(3, epsS: value));
    }

    [Fact]
    public void FiniteParametersThatOverflowFloat32ReturnExplicitFailure()
    {
        var error = Assert.Throws<ArithmeticException>(() => SigmaSchedules.VP(3, betaD: 5000));
        Assert.Contains("non-finite float32", error.Message);
        using var next = SigmaSchedules.VP(3, betaD: 0, betaMin: 0);
        Assert.Equal(new float[] { 0, 0, 0, 0 }, next.data<float>().ToArray());
    }

    private static Tensor Generate(string name, int steps, CancellationToken cancellationToken = default) => name switch
    {
        "karras" => SigmaSchedules.Karras(steps, 0.25, 4, cancellationToken: cancellationToken),
        "exponential" => SigmaSchedules.Exponential(steps, 0.25, 4, cancellationToken),
        "polyexponential" => SigmaSchedules.Polyexponential(steps, 0.25, 4, cancellationToken: cancellationToken),
        "laplace" => SigmaSchedules.Laplace(steps, 0.25, 4, cancellationToken: cancellationToken),
        "vp" => SigmaSchedules.VP(steps, cancellationToken: cancellationToken),
        _ => throw new ArgumentOutOfRangeException(nameof(name))
    };

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, Math.Abs(expected[i]) * 2e-6f + 1e-7f);
    }
}
