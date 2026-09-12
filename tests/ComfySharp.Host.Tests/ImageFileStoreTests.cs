using ComfySharp.Contracts;
using ComfySharp.Storage;

namespace ComfySharp.Host.Tests;

public sealed class ImageFileStoreTests : IDisposable
{
    private readonly string directory = Path.Combine(Path.GetTempPath(), "ComfySharp-image-files-" + Guid.NewGuid().ToString("N"));
    private readonly string outside;
    private readonly ImageFileStore store;
    private readonly List<string> directoryLinks = [];

    public ImageFileStoreTests()
    {
        outside = Path.Combine(directory, "outside"); Directory.CreateDirectory(outside);
        store = new ImageFileStore(Path.Combine(directory, "data"));
    }

    [Fact]
    public async Task Preparation_lists_all_entries_and_reads_independent_output_and_temp_files()
    {
        Assert.Empty(store.PrepareDirectory("output", "album"));
        Assert.Empty(store.PrepareDirectory("temp", "album"));
        var output = new ImageFileDescriptor("image_00001_.png", "album", "output");
        var temp = output with { Type = "temp" };
        await store.WriteAsync(output, new byte[] { 1, 2, 3 });
        await store.WriteAsync(temp, new byte[] { 4, 5 });
        Directory.CreateDirectory(Path.Combine(directory, "data", "output", "album", "image_00077_.png"));
        Assert.Equal(new[] { "image_00001_.png", "image_00077_.png" }, store.PrepareDirectory("output", "album").Order());
        Assert.Equal(new byte[] { 1, 2, 3 }, await Read(output));
        Assert.Equal(new byte[] { 4, 5 }, await Read(temp));
    }

    [Fact]
    public async Task Repeated_name_replaces_and_truncates_existing_file()
    {
        store.PrepareDirectory("output", "");
        var file = new ImageFileDescriptor("image_0_00001_.png", "", "output");
        await store.WriteAsync(file, new byte[] { 1, 2, 3, 4 });
        await store.WriteAsync(file, new byte[] { 9 });
        Assert.Equal(new byte[] { 9 }, await Read(file));
    }

    [Fact]
    public void Absolute_directory_within_media_root_is_allowed_but_outside_and_sibling_roots_are_rejected()
    {
        string media = Path.Combine(directory, "data", "output");
        Assert.Empty(store.PrepareDirectory("output", Path.Combine(media, "album")));
        Assert.True(Directory.Exists(Path.Combine(media, "album")));
        Assert.Throws<ArgumentException>(() => store.PrepareDirectory("output", outside));
        Assert.Throws<ArgumentException>(() => store.PrepareDirectory("output", media + "-sibling"));
    }

    [Fact]
    public void Precancelled_preparation_does_not_create_a_media_directory()
    {
        Assert.Throws<OperationCanceledException>(() => store.PrepareDirectory("temp", "new", new CancellationToken(true)));
        Assert.False(Directory.Exists(Path.Combine(directory, "data", "temp")));
    }

    [Fact]
    public async Task Precancelled_write_preserves_existing_file_and_prior_successful_files()
    {
        store.PrepareDirectory("output", "");
        var first = new ImageFileDescriptor("image_00001_.png", "", "output");
        await store.WriteAsync(first, new byte[] { 7 });
        await Assert.ThrowsAsync<OperationCanceledException>(async () => await store.WriteAsync(first, new byte[] { 9 }, new CancellationToken(true)));
        await Assert.ThrowsAsync<OperationCanceledException>(async () => await store.WriteAsync(first with { Filename = "image_00002_.png" }, new byte[] { 9 }, new CancellationToken(true)));
        Assert.Equal(new byte[] { 7 }, await Read(first));
        Assert.Equal("image_00001_.png", Assert.Single(store.PrepareDirectory("output", "")));
    }

    [Theory]
    [InlineData("../outside")]
    [InlineData("nested/../../outside")]
    [InlineData("../../outside")]
    public void Directory_traversal_cannot_escape_media_root(string subfolder) =>
        Assert.Throws<ArgumentException>(() => store.PrepareDirectory("output", subfolder));

    [Theory]
    [InlineData("unknown")]
    [InlineData("OUTPUT")]
    [InlineData("../outside")]
    public void Unknown_media_types_are_not_redirected_to_output(string type) =>
        Assert.Throws<ArgumentException>(() => store.PrepareDirectory(type, ""));

    [Theory]
    [InlineData("../outside.png")]
    [InlineData(".")]
    [InlineData("")]
    public async Task Descriptor_filename_cannot_contain_a_directory(string name)
    {
        store.PrepareDirectory("output", "");
        var file = new ImageFileDescriptor(name, "", "output");
        await Assert.ThrowsAsync<ArgumentException>(async () => await store.WriteAsync(file, new byte[] { 1 }));
        Assert.Throws<ArgumentException>(() => store.OpenRead(file));
    }

    [Fact]
    public async Task Internal_directory_link_is_supported_but_external_and_dangling_external_links_are_rejected()
    {
        store.PrepareDirectory("output", "actual");
        var media = Path.Combine(directory, "data", "output");
        CreateDirectoryLink(Path.Combine(media, "inside"), Path.Combine(media, "actual"));
        CreateDirectoryLink(Path.Combine(media, "outside"), outside);
        CreateDirectoryLink(Path.Combine(media, "dangling"), Path.Combine(outside, "absent"));
        store.PrepareDirectory("output", "inside");
        var file = new ImageFileDescriptor("image.png", "inside", "output");
        await store.WriteAsync(file, new byte[] { 4 });
        Assert.Equal(new byte[] { 4 }, await File.ReadAllBytesAsync(Path.Combine(media, "actual", "image.png")));
        foreach (string name in new[] { "outside", "dangling" })
            Assert.Throws<ArgumentException>(() => store.PrepareDirectory("output", name));
        Assert.Empty(Directory.GetFileSystemEntries(outside));
    }

    [Fact]
    public async Task Replacing_a_prepared_directory_with_an_external_link_is_detected_at_write()
    {
        store.PrepareDirectory("output", "album");
        string album = Path.Combine(directory, "data", "output", "album");
        Directory.Delete(album); CreateDirectoryLink(album, outside);
        var file = new ImageFileDescriptor("image.png", "album", "output");
        await Assert.ThrowsAsync<ArgumentException>(async () => await store.WriteAsync(file, new byte[] { 1 }));
        Assert.Empty(Directory.GetFileSystemEntries(outside));
    }

    [Fact]
    [Trait("Requires", "SymbolicLinks")]
    public async Task File_links_are_resolved_inside_media_root_and_cannot_escape_on_read_or_write()
    {
        store.PrepareDirectory("output", "");
        string media = Path.Combine(directory, "data", "output");
        await File.WriteAllBytesAsync(Path.Combine(media, "actual.png"), new byte[] { 1 });
        await File.WriteAllBytesAsync(Path.Combine(outside, "keep.png"), new byte[] { 8 });
        File.CreateSymbolicLink(Path.Combine(media, "inside.png"), "actual.png");
        File.CreateSymbolicLink(Path.Combine(media, "outside.png"), Path.Combine(outside, "keep.png"));
        var inside = new ImageFileDescriptor("inside.png", "", "output");
        await store.WriteAsync(inside, new byte[] { 2 }); Assert.Equal(new byte[] { 2 }, await Read(inside));
        var escaped = inside with { Filename = "outside.png" };
        Assert.Throws<ArgumentException>(() => store.OpenRead(escaped));
        await Assert.ThrowsAsync<ArgumentException>(async () => await store.WriteAsync(escaped, new byte[] { 3 }));
        Assert.Equal(new byte[] { 8 }, await File.ReadAllBytesAsync(Path.Combine(outside, "keep.png")));
    }

    private async Task<byte[]> Read(ImageFileDescriptor file)
    {
        await using var stream = store.OpenRead(file);
        using var copy = new MemoryStream(); await stream.CopyToAsync(copy); return copy.ToArray();
    }
    private void CreateDirectoryLink(string path, string target)
    {
        TestDirectoryLink.Create(path, target); directoryLinks.Add(path);
    }
    public void Dispose()
    {
        string temp = Path.GetFullPath(Path.GetTempPath()).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        string target = Path.GetFullPath(directory);
        if (!target.StartsWith(temp, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal) ||
            !Path.GetFileName(target).StartsWith("ComfySharp-image-files-", StringComparison.Ordinal))
            throw new InvalidOperationException("Refusing to clean a directory outside the test workspace.");
        // Remove the links themselves while their targets still exist; never recursively traverse them.
        foreach (string link in directoryLinks.AsEnumerable().Reverse())
        {
            if (OperatingSystem.IsWindows()) Directory.Delete(link); // Junction handle, including dangling targets.
            else File.Delete(link); // unlink the Unix symlink itself, including a dangling directory link.
        }
        Directory.Delete(target, recursive: true);
    }
}
