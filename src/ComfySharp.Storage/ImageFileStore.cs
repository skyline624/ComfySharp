using ComfySharp.Contracts;

namespace ComfySharp.Storage;

/// <summary>Image files under the application's own data directory. Containment is checked
/// after resolving symbolic links/junctions, and again at each write/read. As with the upstream
/// realpath check, this does not lock the directory tree against concurrent external replacement.</summary>
public sealed class ImageFileStore : IImageFileStore
{
    private readonly string root;
    private static StringComparison PathComparison => OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;

    public ImageFileStore(string dataRoot)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(dataRoot);
        root = PhysicalPath(Path.GetFullPath(dataRoot));
        Directory.CreateDirectory(root);
    }

    public IReadOnlyList<string> PrepareDirectory(string type, string subfolder, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string path = ResolveDirectory(type, subfolder);
        Directory.CreateDirectory(path);
        // Recheck after creation before observing contents.
        path = ResolveDirectory(type, subfolder);
        var entries = new List<string>();
        foreach (string entry in Directory.EnumerateFileSystemEntries(path))
        {
            cancellationToken.ThrowIfCancellationRequested();
            entries.Add(Path.GetFileName(entry));
        }
        return entries.AsReadOnly();
    }

    public async ValueTask WriteAsync(ImageFileDescriptor file, ReadOnlyMemory<byte> png, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string path = ResolveFile(file);
        // Source may overwrite a name when %batch_num% prevents matching the prior counter.
        // This is a per-file write, not a transaction over the entire batch.
        await using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read,
            bufferSize: 65536, options: FileOptions.Asynchronous);
        await stream.WriteAsync(png, cancellationToken);
        await stream.FlushAsync(cancellationToken);
    }

    public Stream OpenRead(ImageFileDescriptor file) => new FileStream(ResolveFile(file), FileMode.Open, FileAccess.Read,
        FileShare.Read, bufferSize: 65536, options: FileOptions.Asynchronous | FileOptions.SequentialScan);

    public bool IsFile(ImageFileDescriptor file)
    {
        try { return File.Exists(ResolveFile(file)); }
        catch (ArgumentException) { return false; }
    }

    public async Task WriteAtomicAsync(ImageFileDescriptor file, ReadOnlyMemory<byte> bytes, CancellationToken token)
    {
        string destination = ResolveFile(file);
        var temporary = file with { Filename = ".upload-" + Guid.NewGuid().ToString("N") + ".tmp" };
        string temporaryPath = ResolveFile(temporary);
        try
        {
            await using (var stream = new FileStream(temporaryPath, FileMode.CreateNew, FileAccess.Write, FileShare.None, 65536, FileOptions.Asynchronous))
            {
                await stream.WriteAsync(bytes, token); await stream.FlushAsync(token);
            }
            token.ThrowIfCancellationRequested();
            destination = ResolveFile(file);
            File.Move(temporaryPath, destination, overwrite: true);
        }
        finally { if (File.Exists(temporaryPath)) File.Delete(temporaryPath); }
    }

    private string ResolveFile(ImageFileDescriptor file)
    {
        ArgumentNullException.ThrowIfNull(file);
        if (string.IsNullOrEmpty(file.Filename) || file.Filename is "." or ".." ||
            Path.GetFileName(file.Filename) != file.Filename || Path.IsPathRooted(file.Filename) ||
            file.Filename.Contains('\0') || (OperatingSystem.IsWindows() && file.Filename.Contains(':')))
            throw new ArgumentException("Expected an image filename without a directory or stream suffix.", nameof(file));
        string directory = ResolveDirectory(file.Type, file.Subfolder);
        string path = PhysicalPath(Path.Combine(directory, file.Filename));
        // A file link may target another subfolder, but must remain inside the same media root.
        RequireInside(MediaRoot(file.Type), path);
        return path;
    }

    private string ResolveDirectory(string type, string subfolder)
    {
        ArgumentNullException.ThrowIfNull(subfolder);
        if (subfolder.Contains('\0')) throw new ArgumentException("Invalid image subfolder.", nameof(subfolder));
        string mediaRoot = MediaRoot(type);
        string path = PhysicalPath(Path.Combine(mediaRoot, subfolder));
        RequireInside(mediaRoot, path);
        return path;
    }

    private string MediaRoot(string type)
    {
        if (type is not ("input" or "output" or "temp")) throw new ArgumentException("Image type must be input, output or temp.", nameof(type));
        string path = PhysicalPath(Path.Combine(root, type));
        RequireInside(root, path);
        return path;
    }

    private static void RequireInside(string parent, string path)
    {
        string relative = Path.GetRelativePath(parent, path);
        if (Path.IsPathRooted(relative) || relative == ".." || relative.StartsWith(".." + Path.DirectorySeparatorChar, PathComparison) ||
            (OperatingSystem.IsWindows() && relative.Contains(':')))
            throw new ArgumentException("Image path leaves its media directory.");
    }

    private static string PhysicalPath(string path, int depth = 0)
    {
        if (depth > 63) throw new IOException("Too many symbolic links in image path.");
        string full = Path.GetFullPath(path);
        string current = Path.GetPathRoot(full)!;
        foreach (string part in full[current.Length..].Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, part);
            // LinkTarget returns null for ordinary or absent entries; it also handles a dangling link.
            FileSystemInfo info = Directory.Exists(current) ? new DirectoryInfo(current) : new FileInfo(current);
            string? target = info.LinkTarget;
            if (target is not null)
                current = PhysicalPath(Path.IsPathFullyQualified(target) ? target : Path.Combine(Path.GetDirectoryName(current)!, target), depth + 1);
        }
        return current;
    }
}
