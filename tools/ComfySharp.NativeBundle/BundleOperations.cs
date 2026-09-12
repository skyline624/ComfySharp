using System.Buffers.Binary;
using System.Diagnostics.CodeAnalysis;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace ComfySharp.NativeBundle;

public sealed record BundleFile(string Source, string Destination, long Bytes, string Sha256, string Role, string[]? Aliases = null, string[]? ReplacesSha256 = null);
public sealed record BundleBinding(string Name, string Sha256);
public sealed record BundleRecipe(int SchemaVersion, string Id, string Runtime, string ArchiveSha256, BundleBinding Binding, BundleFile[] Files);

/// <summary>Offline preparation and composition of explicitly pinned native payloads. Never loads archive code.</summary>
public static class BundleOperations
{
    public static JsonSerializerOptions JsonOptions { get; } = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true, UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow };
    private sealed record Stamp(long Bytes, string Sha256);
    private sealed record Tree(string[] Directories, Dictionary<string, Stamp> Files);
    private const string ReceiptName = "comfysharp-native-bundle.json";
    private const long MaxPayloadBytes = 2L * 1024 * 1024 * 1024;
    private static readonly StringComparison PathComparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
    private static string Hash(Stream stream, CancellationToken token)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256); byte[] buffer = new byte[128 * 1024]; int read;
        while ((read = stream.Read(buffer)) != 0) { token.ThrowIfCancellationRequested(); hash.AppendData(buffer, 0, read); }
        token.ThrowIfCancellationRequested(); return Convert.ToHexStringLower(hash.GetHashAndReset());
    }
    private static Stamp Inspect(string path, CancellationToken token)
    {
        RejectLink(path); using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        return new(stream.Length, Hash(stream, token));
    }
    private static bool IsHash(string? value) => value is { Length: 64 } && value.All(Uri.IsHexDigit);
    private static void Require([DoesNotReturnIf(false)] bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
    private static void Relative(string path)
    {
        Require(!string.IsNullOrWhiteSpace(path) && path.Length <= 1024 && !path.StartsWith('/') && !path.Contains('\\') && !path.Any(c => char.IsControl(c) || ":*?\"<>|".Contains(c)), "Unsafe bundle or archive path.");
        Require(path.Split('/').All(p => p.Length != 0 && p is not "." and not ".." && !p.EndsWith(' ') && !p.EndsWith('.')), "Unsafe path component.");
    }
    private static void Leaf(string path) { Relative(path); Require(!path.Contains('/'), "Native image names must be filenames."); }
    private static void RejectLink(string path) => Require((File.GetAttributes(path) & FileAttributes.ReparsePoint) == 0, "Symbolic links and reparse points are not admitted as inputs.");
    private static bool Exists(string path)
    {
        try { _ = File.GetAttributes(path); return true; }
        catch (FileNotFoundException) { return false; }
        catch (DirectoryNotFoundException) { return false; }
    }
    private static bool Within(string path, string root) => path.Equals(root, PathComparison) || path.StartsWith(Path.TrimEndingDirectorySeparator(root) + Path.DirectorySeparatorChar, PathComparison);
    private static void Disjoint(string a, string b) { if (Within(a, b) || Within(b, a)) throw new ArgumentException("Bundle, application and output directories must not overlap."); }
    private static BundleRecipe Load(string path)
    {
        RejectLink(path); Require(new FileInfo(path).Length <= 512 * 1024, "Recipe exceeds the metadata bound.");
        var r = JsonSerializer.Deserialize<BundleRecipe>(File.ReadAllBytes(path), JsonOptions) ?? throw new InvalidDataException("Missing recipe.");
        Require(r.SchemaVersion == 1 && r.Runtime is "linux-x64" or "win-x64" or "osx-arm64", "Unknown bundle schema or runtime.");
        Require(!string.IsNullOrEmpty(r.Id) && r.Id.Length <= 80 && r.Id.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_'), "Invalid bundle identifier.");
        Require(IsHash(r.ArchiveSha256) && r.Binding is not null && IsHash(r.Binding.Sha256), "Invalid archive or binding hash."); Leaf(r.Binding.Name);
        Require(r.Files is { Length: > 0 and <= 32 } && r.Files.All(f => f is not null), "Invalid file allowlist.");
        var sources = new HashSet<string>(StringComparer.Ordinal); var destinations = new HashSet<string>(StringComparer.OrdinalIgnoreCase); var probes = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        long total = 0; bool native = false, notice = false;
        foreach (var f in r.Files)
        {
            Relative(f.Source); Relative(f.Destination);
            Require(sources.Add(f.Source) && destinations.Add(f.Destination), "Duplicate recipe source or destination.");
            Require(f.Bytes > 0 && f.Bytes <= MaxPayloadBytes && IsHash(f.Sha256), "Invalid file size or hash.");
            total = checked(total + f.Bytes); Require(total <= MaxPayloadBytes, "Bundle exceeds its expanded size bound.");
            if (f.Role == "native")
            {
                native = true; Require(f.Destination.StartsWith("native/", StringComparison.Ordinal), "Native payload must be in native/.");
                string name = f.Destination[7..]; Leaf(name);
                foreach (string probe in new[] { name }.Concat(f.Aliases ?? []))
                {
                    Leaf(probe); Require(!probe.Equals(r.Binding.Name, StringComparison.OrdinalIgnoreCase), "The validated TorchSharp binding must remain unchanged.");
                    Require(probes.Add(probe), "Duplicate native filename or alias.");
                }
                Require(f.ReplacesSha256 is { Length: > 0 } && f.ReplacesSha256.All(IsHash), "Native replacement hashes are required.");
            }
            else
            {
                notice = true; Require(f.Role == "notice" && f.Destination.StartsWith("licenses/", StringComparison.Ordinal), "Only native images and license notices are allowed.");
                Require((f.Aliases?.Length ?? 0) == 0 && (f.ReplacesSha256?.Length ?? 0) == 0, "Notices cannot replace native images.");
            }
        }
        Require(native && notice, "Native payload and notices are both required."); return r;
    }
    private static byte[] Receipt(BundleRecipe recipe) => JsonSerializer.SerializeToUtf8Bytes(new { schemaVersion = 1, recipeSha256 = Convert.ToHexStringLower(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(recipe, JsonOptions))), recipe.Runtime, recipe.ArchiveSha256, files = recipe.Files.Select(f => new { f.Destination, f.Bytes, f.Sha256, f.Role }) }, JsonOptions);
    private static string Begin(string output)
    {
        if (Exists(output)) throw new IOException("Destination already exists; no files were replaced.");
        string parent = Path.GetDirectoryName(output) ?? throw new ArgumentException("Output requires a parent directory.");
        Directory.CreateDirectory(parent); string stage = Path.Combine(parent, ".comfysharp-bundle-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(stage); return stage;
    }
    private static void Cleanup(string stage, string output)
    {
        string parent = Path.GetDirectoryName(output)!;
        if (!Within(Path.GetFullPath(stage), parent) || !Path.GetFileName(stage).StartsWith(".comfysharp-bundle-", StringComparison.Ordinal)) throw new InvalidOperationException("Unsafe cleanup target.");
        if (!Exists(stage)) return;
        void Remove(string path)
        {
            var attributes = File.GetAttributes(path); bool directory = (attributes & FileAttributes.Directory) != 0, link = (attributes & FileAttributes.ReparsePoint) != 0;
            if (directory && !link) foreach (string child in Directory.EnumerateFileSystemEntries(path)) Remove(child);
            if (!link && (attributes & FileAttributes.ReadOnly) != 0) File.SetAttributes(path, attributes & ~FileAttributes.ReadOnly);
            if (directory) Directory.Delete(path); else File.Delete(path);
        }
        Remove(stage);
    }
    private static void CheckEnvelope(FileStream stream)
    {
        Require(stream.Length is >= 22 and <= MaxPayloadBytes, "Archive exceeds its size bound or has no ZIP directory.");
        byte[] tail = new byte[(int)Math.Min(stream.Length, 65557)]; stream.Position = stream.Length - tail.Length; stream.ReadExactly(tail);
        for (int i = tail.Length - 22; i >= 0; i--)
        {
            var record = tail.AsSpan(i);
            if (BinaryPrimitives.ReadUInt32LittleEndian(record) != 0x06054b50 || i + 22 + BinaryPrimitives.ReadUInt16LittleEndian(record[20..]) != tail.Length) continue;
            ushort count = BinaryPrimitives.ReadUInt16LittleEndian(record[10..]);
            Require(BinaryPrimitives.ReadUInt32LittleEndian(record[4..]) == 0 && BinaryPrimitives.ReadUInt16LittleEndian(record[8..]) == count, "Split archives are not admitted.");
            uint bytes = BinaryPrimitives.ReadUInt32LittleEndian(record[12..]), offset = BinaryPrimitives.ReadUInt32LittleEndian(record[16..]);
            Require(count <= 50000 && bytes <= 16 * 1024 * 1024 && (long)offset + bytes <= stream.Length - tail.Length + i, "ZIP64, oversized or invalid central directory.");
            stream.Position = 0; return;
        }
        throw new InvalidDataException("ZIP end record is missing.");
    }
    public static void Prepare(string recipePath, string archivePath, string outputPath, CancellationToken token = default)
    {
        token.ThrowIfCancellationRequested(); var recipe = Load(recipePath); string output = Path.GetFullPath(outputPath);
        RejectLink(archivePath); using var archiveFile = new FileStream(archivePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        CheckEnvelope(archiveFile); Require(Hash(archiveFile, token).Equals(recipe.ArchiveSha256, StringComparison.OrdinalIgnoreCase), "Archive hash does not match the pinned recipe."); archiveFile.Position = 0;
        using var zip = new ZipArchive(archiveFile, ZipArchiveMode.Read, true); Require(zip.Entries.Count <= 50000, "Too many archive entries.");
        var entries = new Dictionary<string, ZipArchiveEntry>(StringComparer.Ordinal);
        foreach (var entry in zip.Entries)
        {
            string name = entry.FullName.TrimEnd('/'); Relative(name);
            Require(entries.TryAdd(name, entry), "Duplicate archive entry.");
            Require(((entry.ExternalAttributes >> 16) & 0xf000) != 0xa000, "Symbolic links in archives are not admitted.");
        }
        foreach (var f in recipe.Files) Require(entries.TryGetValue(f.Source, out var entry) && entry.Length == f.Bytes, "Selected entry is missing or has the wrong size.");
        string stage = Begin(output);
        try
        {
            foreach (var f in recipe.Files)
            {
                token.ThrowIfCancellationRequested(); string path = Path.Combine(stage, f.Destination); Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                using (var source = entries[f.Source].Open()) using (var target = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                {
                    byte[] buffer = new byte[128 * 1024]; long copied = 0; int read;
                    while ((read = source.Read(buffer)) != 0) { token.ThrowIfCancellationRequested(); copied = checked(copied + read); Require(copied <= f.Bytes, "Entry exceeds its declared size."); target.Write(buffer, 0, read); }
                    Require(copied == f.Bytes, "Truncated archive entry.");
                }
                Require(Inspect(path, token).Sha256.Equals(f.Sha256, StringComparison.OrdinalIgnoreCase), "Extracted file hash differs.");
            }
            File.WriteAllBytes(Path.Combine(stage, "bundle.json"), Receipt(recipe)); token.ThrowIfCancellationRequested(); Directory.Move(stage, output);
        }
        catch { Cleanup(stage, output); throw; }
    }
    private static Tree Snapshot(string root, CancellationToken token)
    {
        RejectLink(root); var directories = new List<string>(); var files = new Dictionary<string, Stamp>(StringComparer.Ordinal);
        var portablePaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        void Visit(string directory)
        {
            foreach (string path in Directory.EnumerateFileSystemEntries(directory).Order(StringComparer.Ordinal))
            {
                token.ThrowIfCancellationRequested(); RejectLink(path); string relative = Path.GetRelativePath(root, path).Replace('\\', '/'); Relative(relative);
                Require(portablePaths.Add(relative), "Case-ambiguous application or bundle path.");
                if (Directory.Exists(path)) { directories.Add(relative); Visit(path); } else files.Add(relative, Inspect(path, token));
            }
        }
        Visit(root); return new(directories.ToArray(), files);
    }
    private static bool Same(Tree a, Tree b) => a.Directories.SequenceEqual(b.Directories) && a.Files.Count == b.Files.Count && a.Files.All(p => b.Files.TryGetValue(p.Key, out var value) && value == p.Value);
    private static void Copy(string source, string destination, Stamp expected, CancellationToken token)
    {
        token.ThrowIfCancellationRequested(); Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        if (Exists(destination)) File.SetAttributes(destination, File.GetAttributes(destination) & ~FileAttributes.ReadOnly);
        File.Copy(source, destination, true);
        Require(Inspect(destination, token) == expected, "Copied bytes differ from the verified input.");
    }
    public static void Compose(string recipePath, string bundlePath, string applicationPath, string outputPath, CancellationToken token = default)
    {
        token.ThrowIfCancellationRequested(); var recipe = Load(recipePath);
        string bundle = Path.GetFullPath(bundlePath), app = Path.GetFullPath(applicationPath), output = Path.GetFullPath(outputPath);
        Disjoint(bundle, app); Disjoint(bundle, output); Disjoint(app, output);
        var bundleBefore = Snapshot(bundle, token); var expectedBundle = recipe.Files.Select(f => f.Destination).Append("bundle.json").ToHashSet(StringComparer.Ordinal);
        Require(bundleBefore.Files.Keys.ToHashSet(StringComparer.Ordinal).SetEquals(expectedBundle), "Unexpected or missing prepared bundle file.");
        Require(File.ReadAllBytes(Path.Combine(bundle, "bundle.json")).SequenceEqual(Receipt(recipe)), "Prepared bundle receipt does not match the trusted recipe.");
        foreach (var f in recipe.Files) Require(bundleBefore.Files[f.Destination] == new Stamp(f.Bytes, f.Sha256.ToLowerInvariant()), "Prepared payload differs from the recipe.");
        var before = Snapshot(app, token);
        var existingPaths = before.Files.Keys.Concat(before.Directories).ToHashSet(StringComparer.OrdinalIgnoreCase);
        Require(!existingPaths.Contains(ReceiptName), "Application already has a native bundle composition receipt destination.");
        var replacements = new Dictionary<string, BundleFile>(StringComparer.Ordinal); HashSet<string>? nativeDirectories = null;
        foreach (var f in recipe.Files.Where(f => f.Role == "native"))
        {
            string name = f.Destination[7..]; var names = new[] { name }.Concat(f.Aliases ?? []).ToHashSet(StringComparer.Ordinal);
            var probes = before.Files.Where(p => names.Contains(Path.GetFileName(p.Key))).ToArray(); Require(probes.Length != 0, "Application is missing a required native replacement probe.");
            var directories = new HashSet<string>(StringComparer.Ordinal);
            foreach (var probe in probes)
            {
                string directory = Path.GetDirectoryName(probe.Key)!.Replace('\\', '/');
                Require(directory == "" || directory == $"runtimes/{recipe.Runtime}/native", "Native probe is outside the selected runtime layout.");
                Require(probe.Value.Sha256.Equals(f.Sha256, StringComparison.OrdinalIgnoreCase) || (f.ReplacesSha256 ?? []).Contains(probe.Value.Sha256, StringComparer.OrdinalIgnoreCase), "Application contains an unqualified native image."); directories.Add(directory);
            }
            if (nativeDirectories is null) nativeDirectories = directories; else Require(nativeDirectories.SetEquals(directories), "Native payload is split across incompatible directories.");
            foreach (string directory in directories) foreach (string probe in names)
            {
                string destination = directory.Length == 0 ? probe : directory + "/" + probe;
                Require(!existingPaths.Contains(destination) || probes.Any(p => p.Key == destination), "Native alias would overwrite an unqualified or case-conflicting application path.");
                replacements.Add(destination, f);
            }
        }
        foreach (string directory in nativeDirectories!)
        {
            string path = directory.Length == 0 ? recipe.Binding.Name : directory + "/" + recipe.Binding.Name;
            Require(before.Files.TryGetValue(path, out var binding) && binding.Sha256.Equals(recipe.Binding.Sha256, StringComparison.OrdinalIgnoreCase), "TorchSharp binding is missing or not qualified for this recipe.");
        }
        var notices = recipe.Files.Where(f => f.Role == "notice").ToDictionary(f => "third-party/" + recipe.Id + "/" + f.Destination[9..], f => f, StringComparer.Ordinal);
        Require(notices.Keys.All(p => !existingPaths.Contains(p)), "Application notice destination already exists.");
        string stage = Begin(output);
        try
        {
            foreach (string directory in before.Directories) Directory.CreateDirectory(Path.Combine(stage, directory));
            foreach (var file in before.Files) Copy(Path.Combine(app, file.Key), Path.Combine(stage, file.Key), file.Value, token);
            foreach (var replacement in replacements.Concat(notices))
            {
                var f = replacement.Value; Copy(Path.Combine(bundle, f.Destination), Path.Combine(stage, replacement.Key), new(f.Bytes, f.Sha256.ToLowerInvariant()), token);
            }
            Require(Same(before, Snapshot(app, token)) && Same(bundleBefore, Snapshot(bundle, token)), "Inputs changed during composition.");
            File.WriteAllBytes(Path.Combine(stage, ReceiptName), Receipt(recipe)); token.ThrowIfCancellationRequested(); Directory.Move(stage, output);
        }
        catch { Cleanup(stage, output); throw; }
    }
}
