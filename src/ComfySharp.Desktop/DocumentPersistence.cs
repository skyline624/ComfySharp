using ComfySharp.Workflow;

namespace ComfySharp.Desktop;

public static class DocumentPersistence
{
    /// <summary>Edits may continue while writing; only the exact successfully written snapshot becomes saved.</summary>
    public static async Task SaveAsync(WorkflowDocument document, Func<string, Task> writeSnapshot)
    {
        var snapshot = document.ToJson();
        await writeSnapshot(snapshot);
        document.MarkSaved(snapshot);
    }
}
