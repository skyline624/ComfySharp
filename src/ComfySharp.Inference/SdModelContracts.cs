namespace ComfySharp.Inference;

public enum SdAttentionHeadMode { FixedCount, FixedSize }
public enum SdPredictionKind { Epsilon, Velocity }

/// <summary>The plain four-level, two-residual-block SD1/SD2 U-Net topology.
/// Reduced widths describe explicit diagnostic graphs, not additional pretrained architectures.</summary>
public sealed record SdUnetConfig(int BaseChannels, int ContextSize, SdAttentionHeadMode HeadMode,
    int HeadParameter, bool UseLinearProjection)
{
    public static SdUnetConfig Sd15 { get; } = new(320, 768, SdAttentionHeadMode.FixedCount, 8, false);
    public static SdUnetConfig Sd2 { get; } = new(320, 1024, SdAttentionHeadMode.FixedSize, 64, true);
    public const int InputChannels = 4;
    public const int OutputChannels = 4;
    public const int Levels = 4;
    public const int ResidualBlocksPerLevel = 2;
    public const int NormalizationGroups = 32;
    public int TimeEmbeddingChannels => checked(BaseChannels * 4);

    public static int ChannelMultiplier(int level) => level switch
    {
        0 => 1, 1 => 2, 2 or 3 => 4,
        _ => throw new ArgumentOutOfRangeException(nameof(level))
    };

    public int HeadsAtWidth(int width) => HeadMode switch
    {
        SdAttentionHeadMode.FixedCount => HeadParameter,
        SdAttentionHeadMode.FixedSize => width / HeadParameter,
        _ => throw new ArgumentOutOfRangeException(nameof(HeadMode))
    };

    public void Validate()
    {
        if (BaseChannels <= 0 || BaseChannels % NormalizationGroups != 0 || BaseChannels > int.MaxValue / 32)
            throw new ArgumentOutOfRangeException(nameof(BaseChannels), "U-Net base width must be positive, divisible by 32 and fit all channel projections.");
        if (ContextSize <= 0) throw new ArgumentOutOfRangeException(nameof(ContextSize));
        if (!Enum.IsDefined(HeadMode)) throw new ArgumentOutOfRangeException(nameof(HeadMode));
        if (HeadParameter <= 0 || BaseChannels % HeadParameter != 0)
            throw new ArgumentOutOfRangeException(nameof(HeadParameter), "The selected head count or head width must divide every attended channel width.");
    }
}

/// <summary>Classical AutoencoderKL topology with legacy quant/post-quant convolutions for SD image VAEs.</summary>
public sealed record ClassicalVaeConfig(int BaseChannels)
{
    public static ClassicalVaeConfig Stock { get; } = new(128);
    public const int ImageChannels = 3;
    public const int LatentChannels = 4;
    public const int Compression = 8;
    public const int Levels = 4;
    public const int EncoderBlocksPerLevel = 2;
    public const int DecoderBlocksPerLevel = 3;
    public const int NormalizationGroups = 32;
    public static int ChannelMultiplier(int level) => SdUnetConfig.ChannelMultiplier(level);

    public void Validate()
    {
        if (BaseChannels <= 0 || BaseChannels % NormalizationGroups != 0 || BaseChannels > int.MaxValue / 4)
            throw new ArgumentOutOfRangeException(nameof(BaseChannels), "VAE base width must be positive, divisible by 32 and fit all channel levels.");
    }
}
