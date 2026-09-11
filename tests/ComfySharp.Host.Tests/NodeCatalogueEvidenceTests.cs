extern alias catalog;
using System.Text.Json.Nodes;
using Evidence = catalog::ComfySharp.Catalog.NodeCatalogueEvidence;

namespace ComfySharp.Host.Tests;

public sealed class NodeCatalogueEvidenceTests
{
    private const string Commit = "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a";
    private const string Source = "https://github.com/comfy-org/ComfyUI/blob/" + Commit + "/nodes.py#L1";

    private static (JsonObject Manifest, JsonObject Registration, JsonObject Schema) Fixture()
    {
        var manifest = new JsonObject { ["backendCommit"] = Commit, ["capabilities"] = new JsonArray(new JsonObject
        { ["kind"] = "node", ["scope"] = "local", ["name"] = "Example", ["source"] = Source }) };
        var registration = new JsonObject
        {
            ["schemaVersion"] = 1, ["backendCommit"] = Commit, ["registrationComplete"] = false,
            ["unresolvedCases"] = new JsonArray("runtime import not tested"),
            ["coverage"] = new JsonObject { ["loaderModules"] = 1, ["localLoaderModules"] = 1, ["remoteLoaderModules"] = 0,
                ["localRegistrations"] = 1, ["remoteRegistrations"] = 0, ["retainedUnregisteredDeclarations"] = 0 },
            ["modules"] = new JsonArray(new JsonObject { ["module"] = "nodes.py", ["loadOrder"] = 0,
                ["scope"] = "local", ["conditions"] = new JsonArray(), ["source"] = Source, ["candidateCount"] = 1 }),
            ["records"] = new JsonArray(new JsonObject { ["nodeId"] = "Example", ["module"] = "nodes.py", ["sourceClass"] = "Example",
                ["scope"] = "local", ["loadOrder"] = 0, ["registrationState"] = "builtin-candidate",
                ["conditions"] = new JsonArray(), ["source"] = Source, ["classSource"] = Source })
        };
        var schema = new JsonObject
        {
            ["schemaVersion"] = 1, ["backendCommit"] = Commit, ["unresolvedCases"] = new JsonArray(),
            ["records"] = new JsonArray(new JsonObject { ["nodeId"] = "Example", ["module"] = "nodes.py", ["sourceClass"] = "Example",
                ["classSource"] = Source, ["schemaStatus"] = "resolved-static", ["unresolvedFields"] = new JsonArray(),
                ["schemaSources"] = new JsonArray(Source), ["parameterSources"] = new JsonArray() })
        };
        return (manifest, registration, schema);
    }

    [Fact]
    public void ConsistentSourceEvidencePassesWithoutClaimingRuntimeCompleteness()
    {
        var (manifest, registration, schema) = Fixture();
        Assert.Empty(Evidence.Validate(manifest, registration, schema));
        Assert.False(registration["registrationComplete"]!.GetValue<bool>());
    }

    [Theory]
    [InlineData("main")]
    [InlineData("0000000000000000000000000000000000000000")]
    public void EvidenceMustUseTheExactBackendRevision(string revision)
    {
        var (manifest, registration, schema) = Fixture();
        schema["backendCommit"] = revision;
        Assert.NotEmpty(Evidence.Validate(manifest, registration, schema));
    }

    [Theory]
    [InlineData("source", "https://example.org/nodes.py#L1")]
    [InlineData("classSource", "https://github.com/comfy-org/ComfyUI/blob/main/nodes.py#L1")]
    public void RegistrationSourcesCannotDriftOrPointOutsideUpstream(string field, string value)
    {
        var (manifest, registration, schema) = Fixture();
        registration["records"]![0]![field] = value;
        Assert.NotEmpty(Evidence.Validate(manifest, registration, schema));
    }

    [Fact]
    public void MissingModuleSchemaAndManifestRowsAreDetectedIndependently()
    {
        foreach (var remove in new[] { "module", "schema", "manifest" })
        {
            var (manifest, registration, schema) = Fixture();
            (remove == "module" ? registration["modules"] : remove == "schema" ? schema["records"] : manifest["capabilities"])!.AsArray().Clear();
            Assert.NotEmpty(Evidence.Validate(manifest, registration, schema));
        }
    }

    [Fact]
    public void StaleCountsAndIncorrectLoadOrderAreDetected()
    {
        var (manifest, registration, schema) = Fixture();
        registration["coverage"]!["localRegistrations"] = 2;
        registration["records"]![0]!["loadOrder"] = 1;
        var errors = Evidence.Validate(manifest, registration, schema);
        Assert.Contains(errors, e => e.Contains("coverage count"));
        Assert.Contains(errors, e => e.Contains("loader order"));
    }

    [Fact]
    public void UnregisteredDeclarationsDoNotInflateLoaderCandidateCounts()
    {
        var (manifest, registration, schema) = Fixture();
        var retained = registration["records"]![0]!.DeepClone().AsObject();
        retained["nodeId"] = "Unlisted"; retained["sourceClass"] = "Unlisted";
        retained["scope"] = "reference-test"; retained["registrationState"] = "external-custom-reference"; retained["loadOrder"] = null;
        registration["records"]!.AsArray().Add(retained);
        registration["coverage"]!["retainedUnregisteredDeclarations"] = 1;
        manifest["capabilities"]!.AsArray().Add(new JsonObject { ["kind"] = "node", ["scope"] = "reference-test", ["name"] = "Unlisted", ["source"] = Source });
        var retainedSchema = schema["records"]![0]!.DeepClone().AsObject();
        retainedSchema["nodeId"] = "Unlisted"; retainedSchema["sourceClass"] = "Unlisted";
        schema["records"]!.AsArray().Add(retainedSchema);
        Assert.Empty(Evidence.Validate(manifest, registration, schema));
    }

    [Fact]
    public void DynamicExpressionsNeedProvenanceAndCannotBeCalledResolved()
    {
        var (manifest, registration, schema) = Fixture();
        schema["records"]![0]!["unresolvedFields"]!.AsArray().Add(new JsonObject { ["resolution"] = "unresolved", ["expression"] = "runtime.options()" });
        var errors = Evidence.Validate(manifest, registration, schema);
        Assert.Contains(errors, e => e.Contains("Resolved schema"));
        Assert.Contains(errors, e => e.Contains("dynamic expression"));
    }

    [Fact]
    public void RegistrationCompletenessCannotHidePendingCases()
    {
        var (manifest, registration, schema) = Fixture();
        registration["registrationComplete"] = true;
        Assert.Contains(Evidence.Validate(manifest, registration, schema), e => e.Contains("conflicts with unresolved"));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EmbeddedDynamicValuesCannotBeHiddenByClearingTheirSummary(bool claimComplete)
    {
        var (manifest, registration, schema) = Fixture();
        schema["schemaComplete"] = claimComplete;
        schema["records"]![0]!["inputs"] = new JsonObject { ["options"] = new JsonObject
        { ["resolution"] = "unresolved", ["expression"] = "runtime.options()", ["source"] = Source } };
        var errors = Evidence.Validate(manifest, registration, schema);
        Assert.Contains(errors, e => e.Contains("Resolved schema contains dynamic"));
        if (claimComplete) Assert.Contains(errors, e => e.Contains("Schema completeness"));
    }

    [Fact]
    public void ResolvedParameterSourcesMustAlsoBePinned()
    {
        var (manifest, registration, schema) = Fixture();
        schema["records"]![0]!["parameterSources"]!.AsArray().Add(new JsonObject
        { ["path"] = "inputs/value", ["resolution"] = "resolved-static", ["source"] = Source.Replace(Commit, "main") });
        Assert.Contains(Evidence.Validate(manifest, registration, schema), e => e.Contains("parameter inputs/value"));
    }

    [Theory]
    [InlineData("constructor")]
    [InlineData("registrationSources")]
    [InlineData("sharedContract")]
    [InlineData("ioTypeSource")]
    public void NestedAndSharedProvenanceCannotDrift(string location)
    {
        var (manifest, registration, schema) = Fixture();
        var drifted = Source.Replace(Commit, "main");
        if (location == "constructor")
            schema["records"]![0]!["inputs"] = new JsonArray(new JsonObject { ["constructor"] = "io.String.Input", ["source"] = drifted });
        else if (location == "registrationSources")
            schema["records"]![0]!["registrationSources"] = new JsonArray(drifted);
        else
            schema["sharedContract"] = new JsonObject { ["definitions"] = new JsonArray(new JsonObject
            { [location == "ioTypeSource" ? "ioTypeSource" : "source"] = drifted }) };
        Assert.Contains(Evidence.Validate(manifest, registration, schema), e => e.Contains("Unpinned or invalid source"));
    }

    [Fact]
    public void LegacyInputNamedSourceIsNotMistakenForProvenance()
    {
        var (manifest, registration, schema) = Fixture();
        schema["records"]![0]!["inputs"] = new JsonObject { ["required"] = new JsonObject
        { ["source"] = new JsonArray("STRING", new JsonObject { ["default"] = "example" }) } };
        Assert.Empty(Evidence.Validate(manifest, registration, schema));
    }

    [Fact]
    public void ExcludedRemoteNodesMustRemainPresentInManifest()
    {
        var (manifest, registration, schema) = Fixture();
        registration["modules"]![0]!["scope"] = "excluded-remote";
        registration["records"]![0]!["scope"] = "excluded-remote";
        registration["coverage"]!["localLoaderModules"] = 0; registration["coverage"]!["remoteLoaderModules"] = 1;
        registration["coverage"]!["localRegistrations"] = 0; registration["coverage"]!["remoteRegistrations"] = 1;
        // The remaining local row cannot stand in for the excluded remote capability.
        Assert.Contains(Evidence.Validate(manifest, registration, schema), e => e.Contains("manifest capability in its declared scope"));
    }
}
