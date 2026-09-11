namespace ComfySharp.Inference;

/// <summary>Complete legacy-quantized classical image VAE schema, including both directions.</summary>
public static class ClassicalVaeWeightSchema
{
    public static IReadOnlyDictionary<string, IReadOnlyList<long>> Describe(ClassicalVaeConfig config) => Define(config).ReadOnlyShapes;

    internal static ModelWeightSchemaBuilder Define(ClassicalVaeConfig config)
    {
        ArgumentNullException.ThrowIfNull(config); config.Validate();
        var b = new ModelWeightSchemaBuilder();
        long width = config.BaseChannels;
        b.Affine("encoder.conv_in", "encoder.conv_in", width, 3, 3);
        for (int level = 0; level < 4; level++)
        {
            long output = (long)config.BaseChannels * ClassicalVaeConfig.ChannelMultiplier(level);
            for (int block = 0; block < 2; block++)
            {
                Residual($"encoder.down.{level}.block.{block}", $"encoder.down_blocks.{level}.resnets.{block}", width, output);
                width = output;
            }
            if (level < 3) b.Affine($"encoder.down.{level}.downsample.conv", $"encoder.down_blocks.{level}.downsamplers.0.conv", width, width, 3);
        }
        Middle("encoder", width);
        b.Affine("encoder.norm_out", "encoder.conv_norm_out", width);
        b.Affine("encoder.conv_out", "encoder.conv_out", 8, width, 3);
        width = (long)config.BaseChannels * 4;
        b.Affine("decoder.conv_in", "decoder.conv_in", width, 4, 3);
        Middle("decoder", width);
        for (int level = 3; level >= 0; level--)
        {
            long output = (long)config.BaseChannels * ClassicalVaeConfig.ChannelMultiplier(level);
            for (int block = 0; block < 3; block++)
            {
                Residual($"decoder.up.{level}.block.{block}", $"decoder.up_blocks.{3 - level}.resnets.{block}", width, output);
                width = output;
            }
            if (level > 0) b.Affine($"decoder.up.{level}.upsample.conv", $"decoder.up_blocks.{3 - level}.upsamplers.0.conv", width, width, 3);
        }
        b.Affine("decoder.norm_out", "decoder.conv_norm_out", width);
        b.Affine("decoder.conv_out", "decoder.conv_out", 3, width, 3);
        b.Affine("quant_conv", "quant_conv", 8, 8, 1);
        b.Affine("post_quant_conv", "post_quant_conv", 4, 4, 1);
        return b;

        void Residual(string name, string diffusers, long input, long output)
        {
            b.Affine(name + ".norm1", diffusers + ".norm1", input);
            b.Affine(name + ".conv1", diffusers + ".conv1", output, input, 3);
            b.Affine(name + ".norm2", diffusers + ".norm2", output);
            b.Affine(name + ".conv2", diffusers + ".conv2", output, output, 3);
            if (input != output) b.Affine(name + ".nin_shortcut", diffusers + ".conv_shortcut", output, input, 1);
        }

        void Middle(string side, long channels)
        {
            Residual(side + ".mid.block_1", side + ".mid_block.resnets.0", channels, channels);
            Residual(side + ".mid.block_2", side + ".mid_block.resnets.1", channels, channels);
            string name = side + ".mid.attn_1", diffusers = side + ".mid_block.attentions.0";
            b.Affine(name + ".norm", diffusers + ".group_norm", channels);
            foreach (var (canonical, first, second) in new[] { ("q", "to_q", "query"), ("k", "to_k", "key"), ("v", "to_v", "value"), ("proj_out", "to_out.0", "proj_attn") })
            {
                b.Add(name + "." + canonical + ".weight", diffusers + "." + first + ".weight", true, channels, channels, 1, 1);
                b.Add(name + "." + canonical + ".bias", diffusers + "." + first + ".bias", false, channels);
                b.Diffusers.Add(diffusers + "." + second + ".weight", (name + "." + canonical + ".weight", true));
                b.Diffusers.Add(diffusers + "." + second + ".bias", (name + "." + canonical + ".bias", false));
            }
        }
    }
}
