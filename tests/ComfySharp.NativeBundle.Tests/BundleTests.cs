using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using ComfySharp.NativeBundle;
using Xunit;

namespace ComfySharp.NativeBundle.Tests;

public sealed class BundleTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "comfysharp-bundle-" + Guid.NewGuid().ToString("N"));
    private static readonly byte[] Native = Encoding.UTF8.GetBytes("native fixture"), Notice = Encoding.UTF8.GetBytes("notice fixture"), Old = Encoding.UTF8.GetBytes("old native fixture"), Binding = Encoding.UTF8.GetBytes("binding fixture");
    public BundleTests() => Directory.CreateDirectory(root);
    public void Dispose() => Directory.Delete(root, true);
    private string PathFor(string name) => Path.Combine(root, name);
    private static string Hash(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
    private string Fixture(Action<ZipArchive>? extra = null)
    {
        string path = PathFor("source.zip");
        using var zip = ZipFile.Open(path, ZipArchiveMode.Create);
        Add(zip, "torch/lib/libmodern.so", Native); Add(zip, "torch.dist-info/licenses/NOTICE", Notice);
        Add(zip, "torch/__init__.py", Encoding.UTF8.GetBytes("must not be distributed")); extra?.Invoke(zip);
        return path;
    }
    private static void Add(ZipArchive zip, string name, byte[] bytes)
    { using var stream = zip.CreateEntry(name).Open(); stream.Write(bytes); }
    private string Recipe(string archive, Func<BundleRecipe, BundleRecipe>? change = null)
    {
        var recipe = new BundleRecipe(1, "fixture", "linux-x64", Hash(File.ReadAllBytes(archive)),
            new BundleBinding("libLibTorchSharp.so", Hash(Binding)),
            [new("torch/lib/libmodern.so","native/libmodern.so",Native.Length,Hash(Native),"native",["liblegacy.so"],[Hash(Old)]),
             new("torch.dist-info/licenses/NOTICE","licenses/NOTICE",Notice.Length,Hash(Notice),"notice")]);
        string path = PathFor("recipe.json"); File.WriteAllBytes(path, JsonSerializer.SerializeToUtf8Bytes(change is null ? recipe : change(recipe), BundleOperations.JsonOptions)); return path;
    }
    [Fact]
    public void Preparation_extracts_only_verified_native_files_and_notices()
    {
        var archive = Fixture(); var recipe = Recipe(archive); var output = PathFor("bundle");
        BundleOperations.Prepare(recipe, archive, output);
        Assert.Equal(Native, File.ReadAllBytes(Path.Combine(output, "native/libmodern.so")));
        Assert.Equal(Notice, File.ReadAllBytes(Path.Combine(output, "licenses/NOTICE")));
        Assert.Equal(3, Directory.GetFiles(output, "*", SearchOption.AllDirectories).Length);
        Assert.Empty(Directory.GetFiles(output, "*.py", SearchOption.AllDirectories));
    }
    [Theory]
    [InlineData("archive")]
    [InlineData("payload")]
    [InlineData("size")]
    [InlineData("missing")]
    public void Invalid_inputs_leave_no_partial_bundle(string defect)
    {
        var archive = Fixture(); var recipe = Recipe(archive, r => defect switch
        {
            "archive" => r with { ArchiveSha256 = new string('0', 64) },
            "payload" => r with { Files = [r.Files[0] with { Sha256 = new string('0', 64) }, r.Files[1]] },
            "size" => r with { Files = [r.Files[0] with { Bytes = Native.Length + 1 }, r.Files[1]] },
            _ => r with { Files = [r.Files[0] with { Source = "missing.so" }, r.Files[1]] }
        });
        Assert.ThrowsAny<Exception>(() => BundleOperations.Prepare(recipe, archive, PathFor("bundle")));
        Assert.False(Directory.Exists(PathFor("bundle"))); Assert.Empty(Directory.GetDirectories(root));
    }
    [Theory]
    [InlineData("../escape")]
    [InlineData("/absolute")]
    [InlineData("folder\\escape")]
    public void Unsafe_archive_paths_are_rejected_even_when_unselected(string name)
    {
        var archive = Fixture(zip => Add(zip, name, [1])); var recipe = Recipe(archive);
        Assert.Throws<InvalidDataException>(() => BundleOperations.Prepare(recipe, archive, PathFor("bundle")));
    }
    [Fact]
    public void Duplicate_or_symlink_entries_are_rejected()
    {
        var archive = Fixture(zip => Add(zip, "torch/lib/libmodern.so", Native)); var recipe = Recipe(archive);
        Assert.Throws<InvalidDataException>(() => BundleOperations.Prepare(recipe, archive, PathFor("bundle")));
        File.Delete(archive);
        archive = Fixture(zip => { var entry = zip.CreateEntry("link"); entry.ExternalAttributes = unchecked((int)0xa1ff0000); }); recipe = Recipe(archive);
        Assert.Throws<InvalidDataException>(() => BundleOperations.Prepare(recipe, archive, PathFor("bundle")));
    }
    [Fact]
    public void Cancellation_and_existing_destination_preserve_user_files()
    {
        var archive = Fixture(); var recipe = Recipe(archive); var output = PathFor("bundle");
        Assert.Throws<OperationCanceledException>(() => BundleOperations.Prepare(recipe, archive, output, new(true)));
        Directory.CreateDirectory(output); File.WriteAllText(Path.Combine(output, "keep.txt"), "keep");
        Assert.Throws<IOException>(() => BundleOperations.Prepare(recipe, archive, output));
        Assert.Equal("keep", File.ReadAllText(Path.Combine(output, "keep.txt")));
    }
    private string Application(bool flat = false)
    {
        var app = PathFor("app"); var native = flat ? app : Path.Combine(app, "runtimes/linux-x64/native"); Directory.CreateDirectory(native);
        File.WriteAllBytes(Path.Combine(native, "liblegacy.so"), Old); File.WriteAllBytes(Path.Combine(native, "libLibTorchSharp.so"), Binding);
        File.WriteAllText(Path.Combine(app, "program"), "application fixture");
        if (!OperatingSystem.IsWindows()) File.SetUnixFileMode(Path.Combine(app, "program"), UnixFileMode.UserRead | UnixFileMode.UserExecute);
        return app;
    }
    [Fact]
    public void Composition_keeps_binding_application_permissions_and_original_files()
    {
        var archive = Fixture(); var recipe = Recipe(archive); var bundle = PathFor("bundle"); BundleOperations.Prepare(recipe, archive, bundle);
        var app = Application(); var output = PathFor("composed"); BundleOperations.Compose(recipe, bundle, app, output);
        foreach (string name in new[] { "libmodern.so", "liblegacy.so" }) Assert.Equal(Native, File.ReadAllBytes(Path.Combine(output, "runtimes/linux-x64/native", name)));
        Assert.Equal(Old, File.ReadAllBytes(Path.Combine(app, "runtimes/linux-x64/native/liblegacy.so")));
        Assert.Equal(Binding, File.ReadAllBytes(Path.Combine(output, "runtimes/linux-x64/native/libLibTorchSharp.so")));
        Assert.Equal(Notice, File.ReadAllBytes(Path.Combine(output, "third-party/fixture/NOTICE")));
        if (!OperatingSystem.IsWindows()) Assert.Equal(File.GetUnixFileMode(Path.Combine(app, "program")), File.GetUnixFileMode(Path.Combine(output, "program")));
    }
    [Theory]
    [InlineData("bundle")]
    [InlineData("binding")]
    [InlineData("original")]
    public void Composition_rejects_unverified_payloads_before_publishing(string altered)
    {
        var archive = Fixture(); var recipe = Recipe(archive); var bundle = PathFor("bundle"); BundleOperations.Prepare(recipe, archive, bundle); var app = Application();
        string path = altered switch { "bundle" => Path.Combine(bundle, "native/libmodern.so"), "binding" => Path.Combine(app, "runtimes/linux-x64/native/libLibTorchSharp.so"), _ => Path.Combine(app, "runtimes/linux-x64/native/liblegacy.so") };
        File.WriteAllText(path, "changed");
        Assert.Throws<InvalidDataException>(() => BundleOperations.Compose(recipe, bundle, app, PathFor("composed")));
        Assert.False(Directory.Exists(PathFor("composed")));
    }
    [Fact]
    public void Composition_rejects_nested_destinations()
    {
        var archive = Fixture(); var recipe = Recipe(archive); var bundle = PathFor("bundle"); BundleOperations.Prepare(recipe, archive, bundle); var app = Application();
        Assert.Throws<ArgumentException>(() => BundleOperations.Compose(recipe, bundle, app, Path.Combine(app, "child")));
    }

    [Theory]
    [InlineData("../outside")]
    [InlineData("libLibTorchSharp.so")]
    [InlineData("LIBLIBTORCHSHARP.SO")]
    public void Recipe_alias_cannot_escape_or_replace_the_binding(string alias)
    {
        var archive = Fixture(); var recipe = Recipe(archive, r => r with { Files = [r.Files[0] with { Aliases = [alias] }, r.Files[1]] });
        Assert.Throws<InvalidDataException>(() => BundleOperations.Prepare(recipe, archive, PathFor("bundle")));
        Assert.False(Directory.Exists(PathFor("bundle")));
    }

    [Fact]
    public void Flat_publish_layout_composes_and_refuses_overwrite_or_cancellation()
    {
        var archive = Fixture(); var recipe = Recipe(archive); var bundle = PathFor("bundle"); BundleOperations.Prepare(recipe, archive, bundle);
        var app = Application(flat: true); var output = PathFor("composed");
        Assert.Throws<OperationCanceledException>(() => BundleOperations.Compose(recipe, bundle, app, output, new(true)));
        Assert.False(Directory.Exists(output));
        BundleOperations.Compose(recipe, bundle, app, output);
        Assert.Equal(Native, File.ReadAllBytes(Path.Combine(output, "libmodern.so")));
        Assert.Equal(Binding, File.ReadAllBytes(Path.Combine(output, "libLibTorchSharp.so")));
        Assert.Throws<IOException>(() => BundleOperations.Compose(recipe, bundle, app, output));
        Assert.Equal(Old, File.ReadAllBytes(Path.Combine(app, "liblegacy.so")));
    }

    [Theory]
    [InlineData("receipt")]
    [InlineData("extra")]
    public void Prepared_bundle_cannot_add_files_or_change_its_receipt(string defect)
    {
        var archive = Fixture(); var recipe = Recipe(archive); var bundle = PathFor("bundle"); BundleOperations.Prepare(recipe, archive, bundle);
        File.WriteAllText(Path.Combine(bundle, defect == "receipt" ? "bundle.json" : "extra.txt"), "changed");
        Assert.Throws<InvalidDataException>(() => BundleOperations.Compose(recipe, bundle, Application(), PathFor("composed")));
        Assert.False(Directory.Exists(PathFor("composed")));
    }
}
