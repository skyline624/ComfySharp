using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Avalonia.Controls;
using ComfySharp.Workflow;

namespace ComfySharp.Desktop;

public sealed partial class MainWindow
{
    // Explicit opt-in qualification. No model is downloaded and no existing report is overwritten.
    private async Task Sd15SmokeAsync(string reportPath)
    {
        var watch = Stopwatch.StartNew();
        var prompt = Compile(true);
        string[] expectedTypes = ["CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage"];
        foreach (string type in expectedTypes)
            if (!prompt.Any(p => p.Value?["class_type"]?.GetValue<string>() == type))
                throw new InvalidOperationException("SD1.5 smoke document is missing " + type);
        var target = prompt.Single(p => p.Value?["class_type"]?.GetValue<string>() == "SaveImage").Key;
        var snapshot = ActiveEditor.Document.Snapshot(); var ticket = ActiveEditor.BeginSubmission();
        var session = hostSession.Id; var readImage = host.CaptureImageReader(); var images = new JsonArray();
        Console.WriteLine("SD1.5 desktop: submit compiled workflow to Host");
        var accepted = await host.SubmitAsync(prompt, clientId, [target], snapshot);
        var jobId = accepted["prompt_id"]!.GetValue<string>();
        var entry = await WaitForJobAsync(jobId, session, 12000);
        hostSession.Require(session);
        var loaderId = prompt.Single(p => p.Value?["class_type"]?.GetValue<string>() == "CheckpointLoaderSimple").Key;
        var modelInfo = entry["outputs"]?[loaderId]?["comfysharp_model"]?[0]
            ?? throw new InvalidOperationException("Host did not report the actual model placement.");
        if (modelInfo["backend"]?.GetValue<string>() != Program.InferenceDevice ||
            modelInfo["tf32_allowed"]?.GetValue<bool>() != false ||
            modelInfo["component_devices"] is not JsonArray devices || devices.Count != 3 ||
            devices.Any(d => !string.Equals(d?.GetValue<string>(), Program.InferenceDevice, StringComparison.OrdinalIgnoreCase)))
            throw new InvalidOperationException("Host model components do not match the requested inference device.");
        if (!await ActiveEditor.ApplyUiOutputsAsync(entry["outputs"]!.AsObject(), ticket, async (file, token) =>
        {
            byte[] png = await readImage(file, token);
            images.Add(new JsonObject { ["filename"] = file.Filename, ["sha256"] = Convert.ToHexStringLower(SHA256.HashData(png)), ["bytes"] = png.Length });
            return png;
        })) throw new InvalidOperationException("SD1.5 preview was not applied to the submitting document.");
        var view = ActiveEditor.FindControl<Nodify.Avalonia.NodifyEditor>("Canvas")!.ItemsSource!.Cast<NodeView>()
            .Single(n => n.Id.Value == target);
        if (view.ImagePreview?.Image is not { PixelSize.Width: 512, PixelSize.Height: 512 } || images.Count != 1)
            throw new InvalidOperationException("SD1.5 smoke expected one decoded 512x512 desktop preview.");
        var report = new JsonObject
        {
            ["status"] = "ok", ["familyQualified"] = false, ["backend"] = Program.InferenceDevice, ["dtype"] = "Float32",
            ["desktopPreview"] = new JsonArray(512, 512), ["elapsedSeconds"] = watch.Elapsed.TotalSeconds,
            ["prompt"] = prompt.DeepClone(), ["history"] = entry.DeepClone(), ["images"] = images,
            ["workflow"] = snapshot
        };
        await using var output = new FileStream(reportPath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        await using var writer = new StreamWriter(output);
        await writer.WriteAsync(report.ToJsonString(new() { WriteIndented = true }));
        Console.WriteLine("SD1.5 desktop: Host generation and 512x512 preview passed");
    }
}
