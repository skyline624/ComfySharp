using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Microsoft.Data.Sqlite;

namespace ComfySharp.Storage;

public sealed class LocalStore
{
    public string Root { get; }
    private readonly string connectionString;

    public LocalStore(string root)
    {
        Root = Path.GetFullPath(root);
        Directory.CreateDirectory(Root);
        connectionString = new SqliteConnectionStringBuilder { DataSource = Path.Combine(Root, "comfysharp.db"),
            ForeignKeys = true, Pooling = false }.ToString();
        Migrate(1);
    }

    private SqliteConnection Open()
    {
        var connection = new SqliteConnection(connectionString);
        connection.Open();
        return connection;
    }

    public void Migrate(int targetVersion)
    {
        if (targetVersion is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(targetVersion));
        using var connection = Open();
        using var transaction = connection.BeginTransaction();
        using var command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText = "PRAGMA user_version";
        var version = Convert.ToInt32(command.ExecuteScalar());
        if (version > 1) throw new InvalidDataException("Database schema is newer than this application.");
        if (version != targetVersion)
        {
            var resource = $"ComfySharp.Storage.Migrations.001-initial.{(targetVersion == 1 ? "up" : "down")}.sql";
            using var reader = new StreamReader(typeof(LocalStore).Assembly.GetManifestResourceStream(resource)!);
            command.CommandText = reader.ReadToEnd();
            command.ExecuteNonQuery();
            command.CommandText = $"PRAGMA user_version = {targetVersion}";
            command.ExecuteNonQuery();
        }
        transaction.Commit();
    }

    public JsonObject Settings()
    {
        using var connection = Open();
        using var command = connection.CreateCommand();
        command.CommandText = "SELECT key, value FROM settings ORDER BY key";
        using var reader = command.ExecuteReader();
        var settings = new JsonObject();
        while (reader.Read()) settings[reader.GetString(0)] = JsonNode.Parse(reader.GetString(1));
        return settings;
    }

    public void SetSettings(JsonObject settings)
    {
        using var connection = Open();
        using var transaction = connection.BeginTransaction();
        foreach (var (key, value) in settings)
        {
            using var command = connection.CreateCommand();
            command.Transaction = transaction;
            command.CommandText = "INSERT INTO settings(key,value) VALUES($key,$value) ON CONFLICT(key) DO UPDATE SET value=excluded.value";
            command.Parameters.AddWithValue("$key", key);
            command.Parameters.AddWithValue("$value", value?.ToJsonString() ?? "null");
            command.ExecuteNonQuery();
        }
        transaction.Commit();
    }

    public string ResolvePath(string relativePath)
    {
        if (string.IsNullOrWhiteSpace(relativePath) || Path.IsPathRooted(relativePath) || relativePath.Contains(':') || relativePath.Contains('\0'))
            throw new ArgumentException("Expected a relative data path.");
        var path = Path.GetFullPath(Path.Combine(Root, relativePath));
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        if (!path.StartsWith(Root.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar, comparison))
            throw new ArgumentException("Path leaves the ComfySharp data directory.");
        // Existing links could escape even when the lexical path is contained.
        for (var current = path; !string.Equals(current, Root, comparison); current = Path.GetDirectoryName(current)!)
            if ((File.Exists(current) || Directory.Exists(current)) && (File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
                throw new ArgumentException("Linked data paths are not supported.");
        return path;
    }

    public string RegisterAsset(string relativePath, string mediaType, JsonObject? metadata = null)
    {
        var path = ResolvePath(relativePath);
        using var stream = File.OpenRead(path);
        var hash = Convert.ToHexStringLower(SHA256.HashData(stream));
        var id = Guid.NewGuid().ToString();
        using var connection = Open();
        using var command = connection.CreateCommand();
        command.CommandText = "INSERT INTO assets(id,sha256,relative_path,media_type,metadata,created_at) VALUES($id,$hash,$path,$type,$metadata,$date)";
        command.Parameters.AddWithValue("$id", id);
        command.Parameters.AddWithValue("$hash", hash);
        command.Parameters.AddWithValue("$path", relativePath);
        command.Parameters.AddWithValue("$type", mediaType);
        command.Parameters.AddWithValue("$metadata", metadata?.ToJsonString() ?? "{}");
        command.Parameters.AddWithValue("$date", DateTimeOffset.UtcNow.ToString("O"));
        command.ExecuteNonQuery();
        return id;
    }

    public JsonArray Assets()
    {
        using var connection = Open();
        using var command = connection.CreateCommand();
        command.CommandText = "SELECT id,sha256,relative_path,media_type,metadata,missing FROM assets ORDER BY created_at,id";
        using var reader = command.ExecuteReader();
        var result = new JsonArray();
        while (reader.Read()) result.Add(new JsonObject { ["id"] = reader.GetString(0), ["sha256"] = reader.GetString(1),
            ["path"] = reader.GetString(2), ["media_type"] = reader.GetString(3), ["metadata"] = JsonNode.Parse(reader.GetString(4)), ["missing"] = reader.GetBoolean(5) });
        return result;
    }

    public int PruneMissing()
    {
        var absent = Assets().Where(asset => !File.Exists(ResolvePath(asset!["path"]!.GetValue<string>()))).ToArray();
        using var connection = Open();
        using var transaction = connection.BeginTransaction();
        var changed = 0;
        foreach (var asset in absent)
        {
            using var command = connection.CreateCommand();
            command.Transaction = transaction;
            command.CommandText = "UPDATE assets SET missing=1 WHERE id=$id AND missing=0";
            command.Parameters.AddWithValue("$id", asset!["id"]!.GetValue<string>());
            changed += command.ExecuteNonQuery();
        }
        transaction.Commit();
        return changed;
    }
}
