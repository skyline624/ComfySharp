using System.Text.Json.Nodes;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using ComfySharp.Desktop;
using ComfySharp.Workflow;
using Nodify.Avalonia;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public class PreviewEditorTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Pending_host_response_is_discarded_after_restart_or_close(bool close)
    {
        var session = new HostSession(); var original = session.Restart();
        var response = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        var pending = session.ObserveAsync(response.Task, original);
        if (close) session.Close(); else session.Restart();
        response.SetResult("obsolete job or history");
        await Assert.ThrowsAsync<OperationCanceledException>(async () => await pending);
        if (!close) Assert.Equal("current", await session.ObserveAsync(Task.FromResult("current"), session.Id));
    }

    [AvaloniaFact]
    public void Deleted_and_recreated_id_cannot_inherit_an_old_preview_or_inflight_result()
    {
        var editor = new DocumentEditor(); var id = editor.AddNode("PreviewAny");
        var submission = editor.BeginSubmission();
        var output = new JsonObject { [id.Value] = new JsonObject { ["text"] = new JsonArray("obsolete") } };
        Assert.True(editor.ApplyUiOutputs(output, submission));
        editor.Document.Delete(id);
        var replacement = editor.AddNode("PreviewAny"); Assert.Equal(id, replacement);
        Assert.False(editor.ApplyUiOutputs(output, submission));
        var view = Assert.IsType<NodeView>(Assert.Single(editor.FindControl<NodifyEditor>("Canvas")!.ItemsSource!.Cast<object>()));
        Assert.Empty(view.PreviewText);
    }

    [AvaloniaFact]
    public void Earlier_submission_cannot_replace_the_latest_result_and_saving_keeps_validity()
    {
        var editor = new DocumentEditor(); var id = editor.AddNode("PreviewAny");
        var previous = editor.BeginSubmission(); var latest = editor.BeginSubmission();
        editor.Document.MarkSaved();
        JsonObject Output(string text) => new() { [id.Value] = new JsonObject { ["text"] = new JsonArray(text) } };
        Assert.True(editor.ApplyUiOutputs(Output("current"), latest));
        Assert.False(editor.ApplyUiOutputs(Output("old"), previous));
        var view = Assert.IsType<NodeView>(Assert.Single(editor.FindControl<NodifyEditor>("Canvas")!.ItemsSource!.Cast<object>()));
        Assert.Equal("current", view.PreviewText); Assert.False(editor.Document.IsDirty);
    }

    [AvaloniaFact]
    public void PreviewIsVisibleAndNeverBecomesPersistedWidgetData()
    {
        var editor = new DocumentEditor();
        var window = new Window { Content = editor, Width = 1000, Height = 700 }; window.Show();
        try
        {
            var id = editor.AddNode("PreviewAny");
            editor.Document.MarkSaved();
            string before = editor.Document.ToJson();
            editor.ApplyUiOutputs(new JsonObject { [id.Value] = new JsonObject { ["text"] = new JsonArray("tensor([1., 0.])", "second") } });
            Dispatcher.UIThread.RunJobs();
            const string expected = "tensor([1., 0.])\n\nsecond";
            Assert.Contains(window.GetVisualDescendants().OfType<TextBox>(), box => box.IsReadOnly && box.Text == expected);
            Assert.Equal(before, editor.Document.ToJson()); Assert.False(editor.Document.IsDirty);
            Assert.Empty(editor.Document.Nodes[0].Data["widgets_values"]!.AsArray());
            editor.Reload();
            var view = Assert.IsType<NodeView>(Assert.Single(editor.FindControl<NodifyEditor>("Canvas")!.ItemsSource!.Cast<object>()));
            Assert.Equal(expected, view.PreviewText);
            Assert.Empty(PromptCompiler.Compile(editor.Document).Prompt![id.Value]!["inputs"]!.AsObject());
        }
        finally { window.Close(); }
    }

    [AvaloniaTheory]
    [InlineData("KarrasScheduler", "steps,sigma_max,sigma_min,rho")]
    [InlineData("ExponentialScheduler", "steps,sigma_max,sigma_min")]
    [InlineData("PolyexponentialScheduler", "steps,sigma_max,sigma_min,rho")]
    [InlineData("LaplaceScheduler", "steps,sigma_max,sigma_min,mu,beta")]
    [InlineData("VPScheduler", "steps,beta_d,beta_min,eps_s")]
    [InlineData("SplitSigmas", "step")]
    [InlineData("SplitSigmasDenoise", "denoise")]
    [InlineData("FlipSigmas", "")]
    [InlineData("SetFirstSigma", "sigma")]
    [InlineData("ExtendIntermediateSigmas", "steps,start_at_sigma,end_at_sigma,spacing")]
    [InlineData("ManualSigmas", "sigmas")]
    [InlineData("PreviewAny", "")]
    public void NewTemplatesCompileOrderedPersistedWidgets(string type, string names)
    {
        var editor = new DocumentEditor(); var id = editor.AddNode(type);
        var result = PromptCompiler.Compile(editor.Document);
        Assert.True(result.Success, string.Join("; ", result.Diagnostics.Select(d => d.Message)));
        Assert.Equal(names.Split(',', StringSplitOptions.RemoveEmptyEntries), result.Prompt![id.Value]!["inputs"]!.AsObject().Select(p => p.Key));
    }

    [AvaloniaFact]
    public void SigmaConnectionsCompileWithoutCopyingNativeValuesIntoDocuments()
    {
        var editor = new DocumentEditor();
        var schedule = editor.AddNode("KarrasScheduler");
        editor.Document.SetWidgets(schedule, new JsonArray(3, 3.0, 1.0, 1.0));
        var split = editor.AddNode("SplitSigmas");
        editor.Document.SetWidgets(split, new JsonArray(1));
        var preview = editor.AddNode("PreviewAny");
        editor.Document.Connect(schedule, 0, split, 0); editor.Document.Connect(split, 1, preview, 0);
        var restored = WorkflowDocument.Parse(WorkflowDocument.Parse(editor.Document.ToJson()).ToJson());
        var result = PromptCompiler.Compile(restored);
        Assert.True(result.Success);
        Assert.Equal(new JsonArray(split.Value, 1).ToJsonString(), result.Prompt![preview.Value]!["inputs"]!["source"]!.ToJsonString());
        Assert.Equal(new JsonArray(schedule.Value, 0).ToJsonString(), result.Prompt[split.Value]!["inputs"]!["sigmas"]!.ToJsonString());
        Assert.Equal(3, result.Prompt[schedule.Value]!["inputs"]!["steps"]!.GetValue<int>());
        Assert.True(JsonNode.DeepEquals(editor.Document.Snapshot(), restored.Snapshot()));
    }
}
