namespace ComfySharp.Nodes.Tensor;

/// <summary>Read-only shared model catalogue. Never copies or downloads weights.</summary>
public sealed class CheckpointFiles
{
    private readonly string? root;
    public CheckpointFiles(string? modelsDirectory)
    {
        root = string.IsNullOrWhiteSpace(modelsDirectory) ? null : Path.GetFullPath(Path.Combine(modelsDirectory, "checkpoints"));
    }

    public IReadOnlyList<string> Names()
    {
        if (root is null || !Directory.Exists(root)) return [];
        RejectLink(root);
        return Directory.EnumerateFiles(root, "*", new EnumerationOptions
        {
            RecurseSubdirectories = true, AttributesToSkip = FileAttributes.ReparsePoint, IgnoreInaccessible = false
        }).Where(p => p.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            .Select(p => Path.GetRelativePath(root, p).Replace('\\', '/')).Order(StringComparer.Ordinal).ToArray();
    }

    public string Resolve(string name)
    {
        if (root is null) throw new InvalidOperationException("Configure --models-dir or COMFYSHARP_MODELS_DIR with the shared models directory.");
        if (string.IsNullOrWhiteSpace(name) || Path.IsPathRooted(name) || name.Contains(':') || name.Contains('\\'))
            throw new ArgumentException("Checkpoint name must be a relative catalogue name.");
        var parts = name.Split('/');
        if (parts.Any(p => p is "" or "." or ".." || p.EndsWith('.') || p.EndsWith(' ') || p.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0) ||
            !name.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("Only relative safetensors checkpoint names are supported.");
        RejectLink(root);
        string path = root;
        foreach (string part in parts) { path = Path.Combine(path, part); RejectLink(path); }
        if (!File.Exists(path)) throw new FileNotFoundException("Checkpoint is not present in the configured shared catalogue.", name);
        return path;
    }

    private static void RejectLink(string path)
    {
        if ((File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
            throw new ArgumentException("Checkpoint catalogue entries must not traverse symbolic links or junctions.");
    }
}
