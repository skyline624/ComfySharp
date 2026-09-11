extern alias catalog;
using System.Text.Json.Nodes;
using ReleaseQualification = catalog::ComfySharp.Catalog.ReleaseQualification;

namespace ComfySharp.Host.Tests;

public sealed class CatalogTests
{
    private static JsonObject Qualified()
    {
        var platforms = new JsonObject();
        var evidence = new JsonArray();
        foreach (var platform in ReleaseQualification.RequiredPlatforms)
        {
            platforms[platform] = true;
            evidence.Add(new JsonObject { ["platform"] = platform, ["hardware"] = "test fixture hardware",
                ["scenario"] = "contract:fixture", ["artifact"] = "fixtures/run-result.json", ["passed"] = true });
        }
        return new JsonObject { ["catalogueComplete"] = true, ["capabilities"] = new JsonArray(new JsonObject
        {
            ["id"] = "fixture", ["scope"] = "local", ["implementation"] = "implemented",
            ["unitTests"] = true, ["realWorkflow"] = true, ["platforms"] = platforms, ["evidence"] = evidence
        }) };
    }

    [Fact]
    public void AllSixPlatformsWithStructuredEvidencePassStructuralGate() => Assert.Empty(ReleaseQualification.Validate(Qualified()));

    [Theory]
    [InlineData("{}")]
    [InlineData("{\"win-x64-cpu\":true}")]
    [InlineData("null")]
    public void MissingPlatformsFailEvenWithAllOtherClaimsTrue(string json)
    {
        var manifest = Qualified();
        manifest["capabilities"]![0]!["platforms"] = JsonNode.Parse(json);
        Assert.NotEmpty(ReleaseQualification.Validate(manifest));
    }

    [Fact]
    public void EveryRequiredPlatformAndItsEvidenceIsMandatory()
    {
        foreach (var platform in ReleaseQualification.RequiredPlatforms)
        {
            var manifest = Qualified();
            manifest["capabilities"]![0]!["platforms"]!.AsObject().Remove(platform);
            Assert.NotEmpty(ReleaseQualification.Validate(manifest));
            manifest = Qualified();
            var evidence = manifest["capabilities"]![0]!["evidence"]!.AsArray();
            evidence.Remove(evidence.Single(e => e!["platform"]!.GetValue<string>() == platform));
            Assert.NotEmpty(ReleaseQualification.Validate(manifest));
        }
    }

    [Theory]
    [InlineData("[]")]
    [InlineData("[{}]")]
    [InlineData("[true]")]
    [InlineData("[\"run passed\"]")]
    [InlineData("null")]
    public void ArbitraryEvidenceDoesNotQualify(string json)
    {
        var manifest = Qualified();
        manifest["capabilities"]![0]!["evidence"] = JsonNode.Parse(json);
        Assert.NotEmpty(ReleaseQualification.Validate(manifest));
    }

    [Fact]
    public void CompletenessMustBeTrue()
    {
        foreach (var value in new JsonNode?[] { null, JsonValue.Create(false), JsonValue.Create("true") })
        {
            var manifest = Qualified();
            manifest["catalogueComplete"] = value;
            Assert.NotEmpty(ReleaseQualification.Validate(manifest));
        }
    }

    [Theory]
    [InlineData("hardware")]
    [InlineData("scenario")]
    [InlineData("artifact")]
    [InlineData("passed")]
    [InlineData("platform")]
    public void EvidenceRequiresAllRunFields(string field)
    {
        var manifest = Qualified();
        manifest["capabilities"]![0]!["evidence"]![0]!.AsObject().Remove(field);
        Assert.NotEmpty(ReleaseQualification.Validate(manifest));
    }

    [Theory]
    [InlineData("false")]
    [InlineData("\"true\"")]
    [InlineData("null")]
    public void PlatformClaimsMustBeBooleanTrue(string json)
    {
        var manifest = Qualified();
        manifest["capabilities"]![0]!["platforms"]!["osx-arm64-mps"] = JsonNode.Parse(json);
        Assert.NotEmpty(ReleaseQualification.Validate(manifest));
    }
}
