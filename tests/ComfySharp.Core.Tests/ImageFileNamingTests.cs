using System.Globalization;
using System.Numerics;
using ComfySharp.Contracts;
using ComfySharp.Nodes;
using Xunit;

namespace ComfySharp.Core.Tests;

public sealed class ImageFileNamingTests
{
    private static readonly DateTimeOffset Time = new(2026, 9, 12, 7, 8, 9, TimeSpan.FromHours(2));

    [Theory]
    [InlineData("ComfyUI_00009_.png", "10")]
    [InlineData("ComfyUI_00021.txt", "22")]
    [InlineData("ComfyUI_-4_.png", "-3")]
    [InlineData("ComfyUI_+41.any", "42")]
    [InlineData("ComfyUI_ ٧٢ _.png", "73")]
    [InlineData("ComfyUI_１２３_.png", "124")]
    [InlineData("ComfyUI_𝟡_.png", "10")]
    [InlineData("ComfyUI_9999999999999999999999999999999999999_.png", "10000000000000000000000000000000000000")]
    [InlineData("ComfyUI_no-number.png", "1")]
    [InlineData("ComfyUI_7_00099_.png", "8")]
    [InlineData("ComfyUI_7.999.png", "8")]
    [InlineData("ComfyUI_².png", "1")]
    [InlineData("ComfyUI_\u001c9.png", "1")]
    [InlineData("ComfyUI_extra_99_.png", "1")]
    [InlineData("ComfyUIx_999_.png", "1")]
    [InlineData("ComfyUI", "1")]
    public void Counter_scan_matches_prefix_and_integer_prefix_without_extension_filter(string entry, string expected)
    {
        var plan = ImageFileNaming.Prepare(new DirectoryEntries(entry), "output", "ComfyUI", 8, 4, Time);
        Assert.Equal(BigInteger.Parse(expected, CultureInfo.InvariantCulture), plan.Counter);
    }

    [Fact]
    public void Empty_directory_starts_at_one_and_negative_counters_use_sign_aware_padding()
    {
        var plan = ImageFileNaming.Prepare(new DirectoryEntries(), "output", "ComfyUI", 8, 4, Time);
        Assert.Equal(new ImageFileDescriptor("ComfyUI_00001_.png", "", "output"), plan.Frame("output", 0));
        var negative = ImageFileNaming.Prepare(new DirectoryEntries("ComfyUI_-4_.png", "ComfyUI_-2_.png"), "temp", "ComfyUI", 8, 4, Time);
        Assert.Equal("ComfyUI_-0001_.png", negative.Frame("temp", 0).Filename);
        Assert.Equal("ComfyUI_00000_.png", negative.Frame("temp", 1).Filename);
    }

    [Fact]
    public void Variables_expand_in_source_order_and_unknown_frontend_formats_remain_literal()
    {
        var entries = new DirectoryEntries();
        var plan = ImageFileNaming.Prepare(entries, "output", "set/%year%-%month%-%day%/%width%x%height%_%hour%-%minute%-%second%_%date:fmt%", 32, 17, Time);
        Assert.Equal(Path.Combine("set", "2026-09-12"), plan.Subfolder);
        Assert.Equal("32x17_07-08-09_%date:fmt%", plan.Filename);
        Assert.Equal(plan.Subfolder, entries.Subfolder);
        Assert.Equal("output", entries.Type);
    }

    [Fact]
    public void Batch_placeholder_is_substituted_after_scanning_and_can_reuse_a_previous_name()
    {
        var plan = ImageFileNaming.Prepare(new DirectoryEntries("image_0_00001_.png", "image_1_00002_.png"),
            "output", "image_%batch_num%", 8, 4, Time);
        Assert.Equal(BigInteger.One, plan.Counter);
        Assert.Equal("image_0_00001_.png", plan.Frame("output", 0).Filename);
        Assert.Equal("image_1_00002_.png", plan.Frame("output", 1).Filename);
    }

    [Fact]
    public void Scan_slice_length_uses_unicode_scalars_not_utf16_units()
    {
        var plan = ImageFileNaming.Prepare(new DirectoryEntries("😀猫_00032_.png"), "output", "😀猫", 1, 1, Time);
        Assert.Equal(new BigInteger(33), plan.Counter);
        Assert.Equal("😀猫_00033_.png", plan.Frame("output", 0).Filename);
    }

    [Fact]
    public void Case_matching_uses_native_platform_normcase_rules()
    {
        var plan = ImageFileNaming.Prepare(new DirectoryEntries("IMAGE_00042_.png"), "output", "image", 1, 1, Time);
        Assert.Equal(new BigInteger(OperatingSystem.IsWindows() ? 43 : 1), plan.Counter);
        // Full lowercase can expand a scalar; OrdinalIgnoreCase would not implement Python normcase.
        var expanded = ImageFileNaming.Prepare(new DirectoryEntries("İ_00008_.png"), "output", "İ", 1, 1, Time);
        Assert.Equal(new BigInteger(9), expanded.Counter);
        var finalSigma = ImageFileNaming.Prepare(new DirectoryEntries("ς_00008_.png"), "output", "Σ", 1, 1, Time);
        Assert.Equal(BigInteger.One, finalSigma.Counter);
    }

    [Theory]
    [InlineData("", ".", "")]
    [InlineData("a/./b/../image", "image", "a")]
    [InlineData("a/", "a", "")]
    public void Normalization_preserves_source_basename_and_trailing_separator_scan(string prefix, string filename, string subfolder)
    {
        var plan = ImageFileNaming.Prepare(new DirectoryEntries(filename + "_00099_.png"), "output", prefix, 1, 1, Time);
        Assert.Equal(filename, plan.Filename); Assert.Equal(subfolder, plan.Subfolder);
        Assert.Equal(new BigInteger(prefix.EndsWith('/') || prefix.Length == 0 ? 1 : 100), plan.Counter);
    }

    [Fact]
    public void Cancellation_prevents_directory_preparation()
    {
        var entries = new DirectoryEntries();
        Assert.Throws<OperationCanceledException>(() => ImageFileNaming.Prepare(entries, "output", "image", 1, 1, Time, new CancellationToken(true)));
        Assert.Null(entries.Type);
    }

    private sealed class DirectoryEntries(params string[] entries) : IImageFileStore
    {
        public string? Subfolder { get; private set; }
        public string? Type { get; private set; }
        public IReadOnlyList<string> PrepareDirectory(string type, string subfolder, CancellationToken cancellationToken = default)
        { Type = type; Subfolder = subfolder; return entries; }
        public ValueTask WriteAsync(ImageFileDescriptor file, ReadOnlyMemory<byte> png, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException("This fixture only observes directory preparation.");
    }
}
