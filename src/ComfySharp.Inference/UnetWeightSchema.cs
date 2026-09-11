namespace ComfySharp.Inference;

/// <summary>Complete plain SD1/SD2 U-Net names and shapes. Pure metadata; no native initialization.</summary>
public static class UnetWeightSchema
{
    public static IReadOnlyDictionary<string, IReadOnlyList<long>> Describe(SdUnetConfig config) => Define(config).ReadOnlyShapes;

    internal static ModelWeightSchemaBuilder Define(SdUnetConfig config)
    {
        ArgumentNullException.ThrowIfNull(config); config.Validate();
        var b = new ModelWeightSchemaBuilder();
        long width = config.BaseChannels, time = config.TimeEmbeddingChannels;
        b.Affine("time_embed.0", "time_embedding.linear_1", time, width);
        b.Affine("time_embed.2", "time_embedding.linear_2", time, time);
        b.Affine("input_blocks.0.0", "conv_in", width, 4, 3);
        var skips = new Stack<long>(); skips.Push(width);
        int inputIndex = 1;
        for (int level = 0; level < 4; level++)
        {
            long output = (long)config.BaseChannels * SdUnetConfig.ChannelMultiplier(level);
            for (int block = 0; block < 2; block++)
            {
                Residual($"input_blocks.{inputIndex}.0", $"down_blocks.{level}.resnets.{block}", width, output);
                width = output;
                if (level < 3) Spatial($"input_blocks.{inputIndex}.1", $"down_blocks.{level}.attentions.{block}", width);
                skips.Push(width); inputIndex++;
            }
            if (level < 3)
            {
                b.Affine($"input_blocks.{inputIndex}.0.op", $"down_blocks.{level}.downsamplers.0.conv", width, width, 3);
                skips.Push(width); inputIndex++;
            }
        }
        Residual("middle_block.0", "mid_block.resnets.0", width, width);
        Spatial("middle_block.1", "mid_block.attentions.0", width);
        Residual("middle_block.2", "mid_block.resnets.1", width, width);
        int outputIndex = 0;
        for (int level = 3; level >= 0; level--)
        {
            long output = (long)config.BaseChannels * SdUnetConfig.ChannelMultiplier(level);
            for (int block = 0; block < 3; block++)
            {
                Residual($"output_blocks.{outputIndex}.0", $"up_blocks.{3 - level}.resnets.{block}", width + skips.Pop(), output);
                width = output;
                if (level < 3) Spatial($"output_blocks.{outputIndex}.1", $"up_blocks.{3 - level}.attentions.{block}", width);
                if (level > 0 && block == 2)
                    b.Affine($"output_blocks.{outputIndex}.{(level == 3 ? 1 : 2)}.conv", $"up_blocks.{3 - level}.upsamplers.0.conv", width, width, 3);
                outputIndex++;
            }
        }
        b.Affine("out.0", "conv_norm_out", width);
        b.Affine("out.2", "conv_out", 4, width, 3);
        return b;

        void Residual(string name, string diffusers, long input, long output)
        {
            b.Affine(name + ".in_layers.0", diffusers + ".norm1", input);
            b.Affine(name + ".in_layers.2", diffusers + ".conv1", output, input, 3);
            b.Affine(name + ".emb_layers.1", diffusers + ".time_emb_proj", output, time);
            b.Affine(name + ".out_layers.0", diffusers + ".norm2", output);
            b.Affine(name + ".out_layers.3", diffusers + ".conv2", output, output, 3);
            if (input != output) b.Affine(name + ".skip_connection", diffusers + ".conv_shortcut", output, input, 1);
        }

        void Spatial(string name, string diffusers, long channels)
        {
            b.Affine(name + ".norm", diffusers + ".norm", channels);
            foreach (string projection in new[] { "proj_in", "proj_out" })
                b.Affine(name + "." + projection, diffusers + "." + projection, channels, channels, config.UseLinearProjection ? 0 : 1);
            string t = name + ".transformer_blocks.0", d = diffusers + ".transformer_blocks.0";
            for (int norm = 1; norm <= 3; norm++) b.Affine(t + $".norm{norm}", d + $".norm{norm}", channels);
            for (int attention = 1; attention <= 2; attention++)
            {
                foreach (string projection in new[] { "q", "k", "v" })
                    b.Affine(t + $".attn{attention}.to_{projection}", d + $".attn{attention}.to_{projection}", channels,
                        attention == 1 || projection == "q" ? channels : config.ContextSize, bias: false);
                b.Affine(t + $".attn{attention}.to_out.0", d + $".attn{attention}.to_out.0", channels, channels);
            }
            b.Affine(t + ".ff.net.0.proj", d + ".ff.net.0.proj", channels * 8, channels);
            b.Affine(t + ".ff.net.2", d + ".ff.net.2", channels, channels * 4);
        }
    }
}
