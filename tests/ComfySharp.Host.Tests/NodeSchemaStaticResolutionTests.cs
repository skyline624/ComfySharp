extern alias catalog;
using System.Text.Json.Nodes;
using Evidence = catalog::ComfySharp.Catalog.NodeCatalogueEvidence;
using Resolution = catalog::ComfySharp.Catalog.NodeSchemaStaticResolution;

namespace ComfySharp.Host.Tests;

public sealed class NodeSchemaStaticResolutionTests
{
    [Fact]
    public void PublishedPlanAndAllAppliedOccurrencesHaveTheReviewedIdentity()
    {
        var (schemas, plan) = Documents();
        Assert.Equal(Resolution.PlanSha256, Resolution.PlanDigest(plan));
        Assert.Equal(17, plan["rules"]!.AsArray().Count);
        Assert.Equal(55, plan["occurrences"]!.AsArray().Count);
        Assert.Empty(Resolution.ValidateApplied(schemas, plan));
        Assert.Empty(Evidence.Validate(Read("manifest.json"), Read("node-registration.json"), schemas, plan));
        Assert.Equal(1171, Read("manifest.json")["capabilities"]!.AsArray().Count);
        Assert.False(schemas["schemaComplete"]!.GetValue<bool>());
        Assert.False(schemas["catalogueComplete"]!.GetValue<bool>());
    }

    [Fact]
    public void ReviewedLiteralValuesPreserveStringEnumMeaningAndDeclarationOrder()
    {
        var (_, plan) = Documents();
        var rules = plan["rules"]!.AsArray().OfType<JsonObject>().ToDictionary(r => r["expression"]!.GetValue<string>());
        Assert.Equal("file_upload", rules["comfy_api.latest.IO.UploadType.model"]["value"]!.GetValue<string>());
        Assert.Equal("PROMPT", rules["comfy_api.latest.io.Hidden.prompt"]["value"]!.GetValue<string>());
        Assert.Equal("EXTRA_PNGINFO", rules["comfy_api.latest.IO.Hidden.extra_pnginfo"]["value"]!.GetValue<string>());
        Assert.Equal("STRING", rules["comfy.comfy_types.IO.STRING"]["value"]!.GetValue<string>());
        Assert.Equal("CLIP", rules["comfy.comfy_types.IO.CLIP"]["value"]!.GetValue<string>());
        Assert.Equal("CONDITIONING", rules["comfy.comfy_types.IO.CONDITIONING"]["value"]!.GetValue<string>());
        Assert.Equal(new[] { "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal" },
            Strings(rules["list(SCHEDULER_HANDLERS)"]["value"]!));
        Assert.Equal(new[] { "ADD", "CLEAR", "DARKEN", "DST", "DST_ATOP", "DST_IN", "DST_OUT", "DST_OVER", "LIGHTEN", "MULTIPLY", "OVERLAY", "SCREEN", "SRC", "SRC_ATOP", "SRC_IN", "SRC_OUT", "SRC_OVER", "XOR" },
            Strings(rules["[mode.name for mode in PorterDuffMode]"]["value"]!));
        Assert.Equal("DST", rules["PorterDuffMode.DST.name"]["value"]!.GetValue<string>());
    }

    [Fact]
    public void PureTransformationChangesExactlyTheReviewed55OccurrencesAnd26Statuses()
    {
        var (published, plan) = Documents();
        var original = Undo(published, plan);
        Assert.Empty(Resolution.ValidateAccounting(original));
        Assert.Equal(281, original["coverage"]!["unresolvedExpressionCount"]!.GetValue<int>());
        Assert.Equal(130, original["coverage"]!["statuses"]!["partial-static"]!.GetValue<int>());
        string before = original.ToJsonString(), planBefore = plan.ToJsonString();
        var result = Resolution.Apply(original, plan);
        Assert.Equal(before, original.ToJsonString());
        Assert.Equal(planBefore, plan.ToJsonString());
        Assert.True(JsonNode.DeepEquals(published, result));
        Assert.Equal(663, result["records"]!.AsArray().Count);
        Assert.Equal(226, result["coverage"]!["unresolvedExpressionCount"]!.GetValue<int>());
        Assert.Equal(559, result["coverage"]!["statuses"]!["resolved-static"]!.GetValue<int>());
        Assert.Equal(104, result["coverage"]!["statuses"]!["partial-static"]!.GetValue<int>());
        var oldRecords = Records(original).ToDictionary(Key);
        int touched = 0, completed = 0;
        foreach (var record in Records(result))
        {
            var old = oldRecords[Key(record)];
            if (record["resolvedFields"] is not JsonArray) Assert.True(JsonNode.DeepEquals(old, record));
            else
            {
                touched++;
                if (record["schemaStatus"]!.GetValue<string>() == "resolved-static") completed++;
                var untouchedMirrors = old["unresolvedFields"]!.AsArray().Where(f => !plan["occurrences"]!.AsArray().OfType<JsonObject>()
                    .Any(o => Key(o) == Key(record) && o["path"]!.GetValue<string>() == f!["path"]!.GetValue<string>())).ToArray();
                Assert.Equal(untouchedMirrors.Length, record["unresolvedFields"]!.AsArray().Count);
                foreach (var mirror in untouchedMirrors)
                    Assert.Contains(record["unresolvedFields"]!.AsArray(), actual => JsonNode.DeepEquals(mirror, actual));
            }
        }
        Assert.Equal(35, touched); Assert.Equal(26, completed);
        Assert.Throws<InvalidOperationException>(() => Resolution.Apply(result, plan));
    }

    [Theory]
    [InlineData("backend")]
    [InlineData("definition-source")]
    [InlineData("alias-source")]
    [InlineData("value")]
    [InlineData("ordered-list")]
    [InlineData("missing-rule")]
    [InlineData("unknown-expression")]
    [InlineData("path")]
    public void AnyChangeToTheReviewedPlanIsRejected(string mutation)
    {
        var (schemas, plan) = Documents();
        switch (mutation)
        {
            case "backend": plan["backendCommit"] = new string('0', 40); break;
            case "definition-source": plan["rules"]![0]!["definitionSource"] = "https://github.com/comfy-org/ComfyUI/blob/main/nodes.py#L1"; break;
            case "alias-source": plan["rules"]!.AsArray().OfType<JsonObject>().First(r => r["sharedContract"] is not null)["sources"]![0] = "https://example.org/alias.py"; break;
            case "value": plan["rules"]![0]!["value"] = "guessed"; break;
            case "ordered-list":
                var list = plan["rules"]!.AsArray().OfType<JsonObject>().First(r => r["kind"]!.GetValue<string>() == "ordered-dictionary-keys")["value"]!.AsArray();
                list[0] = "normal"; break;
            case "missing-rule": plan["rules"]!.AsArray().RemoveAt(0); break;
            case "unknown-expression": plan["occurrences"]![0]!["expression"] = "comfy_api.latest.io.Image"; break;
            case "path": plan["occurrences"]![0]!["path"] = "inputs/missing"; break;
        }
        Assert.NotEmpty(Resolution.ValidateApplied(schemas, plan));
    }

    [Theory]
    [InlineData("value")]
    [InlineData("source")]
    [InlineData("missing-evidence")]
    [InlineData("duplicate-evidence")]
    [InlineData("shared-enum")]
    [InlineData("marker")]
    public void AppliedValuesAndProvenanceCannotDriftIndependently(string mutation)
    {
        var (schemas, plan) = Documents(); var occurrence = plan["occurrences"]![0]!.AsObject();
        var record = Records(schemas).Single(r => Key(r) == Key(occurrence));
        switch (mutation)
        {
            case "value": Set(record, occurrence["path"]!.GetValue<string>(), JsonValue.Create("wrong")!); break;
            case "source": record["resolvedFields"]![0]!["source"] = "https://example.org/definition"; break;
            case "missing-evidence": record["resolvedFields"]!.AsArray().RemoveAt(0); break;
            case "duplicate-evidence": record["resolvedFields"]!.AsArray().Add(record["resolvedFields"]![0]!.DeepClone()); break;
            case "shared-enum": schemas["sharedContract"]!["definitions"]!.AsArray().OfType<JsonObject>().Single(d => d["name"]!.GetValue<string>() == "Hidden")["enumValues"]!["prompt"] = "prompt"; break;
            case "marker": schemas["staticResolution"]!["occurrences"] = 54; break;
        }
        Assert.NotEmpty(Resolution.ValidateApplied(schemas, plan));
    }

    [Theory]
    [InlineData("coverage-total")]
    [InlineData("coverage-status")]
    [InlineData("summary")]
    [InlineData("mirror-path")]
    [InlineData("mirror-expression")]
    [InlineData("mirror-removed")]
    [InlineData("parameter-status")]
    [InlineData("invalid-status")]
    [InlineData("schema-complete")]
    public void AccountingRejectsStaleSummariesWithoutDoubleCountingMirrors(string mutation)
    {
        var (schemas, _) = Documents();
        var partial = Records(schemas).First(r => r["unresolvedFields"]!.AsArray().Count > 0);
        switch (mutation)
        {
            case "coverage-total": schemas["coverage"]!["unresolvedExpressionCount"] = 452; break;
            case "coverage-status": schemas["coverage"]!["statuses"]!["resolved-static"] = 560; break;
            case "summary": schemas["unresolvedCases"]!.AsArray().RemoveAt(0); break;
            case "mirror-path": partial["unresolvedFields"]![0]!["path"] = "inputs/missing"; break;
            case "mirror-expression": partial["unresolvedFields"]![0]!["expression"] = "different.expression"; break;
            case "mirror-removed": partial["unresolvedFields"]!.AsArray().RemoveAt(0); break;
            case "parameter-status": Records(schemas).First(r => r["parameterSources"]!.AsArray().Count > 0)["parameterSources"]![0]!["resolution"] = "partial-static"; break;
            case "invalid-status": partial["schemaStatus"] = "made-up-status"; break;
            case "schema-complete": schemas["schemaComplete"] = true; break;
        }
        Assert.NotEmpty(Resolution.ValidateAccounting(schemas));
    }

    [Fact]
    public void AFailureAtTheLastOccurrenceDoesNotPublishEarlierChanges()
    {
        var (schemas, plan) = Documents(); var original = Undo(schemas, plan);
        var last = plan["occurrences"]!.AsArray().Last()!.AsObject();
        var record = Records(original).Single(r => Key(r) == Key(last));
        var value = At(record, last["path"]!.GetValue<string>())!.AsObject(); value["expression"] = "unreviewed.expression";
        record["unresolvedFields"]!.AsArray().OfType<JsonObject>().Single(f => f["path"]!.GetValue<string>() == last["path"]!.GetValue<string>())["expression"] = "unreviewed.expression";
        Assert.Empty(Resolution.ValidateAccounting(original));
        string before = original.ToJsonString();
        Assert.Throws<InvalidOperationException>(() => Resolution.Apply(original, plan));
        Assert.Equal(before, original.ToJsonString());
        Assert.False(original.ContainsKey("staticResolution"));
    }

    [Fact]
    public void EvidenceCannotBeValidatedWithoutItsReviewedPlan()
    {
        var (schemas, _) = Documents();
        Assert.Contains(Evidence.Validate(Read("manifest.json"), Read("node-registration.json"), schemas),
            error => error.Contains("Missing reviewed static resolution plan"));
    }

    [Theory]
    [InlineData("schemaComplete")]
    [InlineData("catalogueComplete")]
    public void StandaloneTransformationCannotPromoteCompleteness(string field)
    {
        var (schemas, plan) = Documents(); var original = Undo(schemas, plan);
        original[field] = true;
        string before = original.ToJsonString();
        Assert.Throws<InvalidOperationException>(() => Resolution.Apply(original, plan));
        Assert.Equal(before, original.ToJsonString());
        schemas[field] = true;
        Assert.NotEmpty(Resolution.ValidateApplied(schemas, plan));
    }

    [Theory]
    [InlineData("absent")]
    [InlineData("null")]
    [InlineData("non-array")]
    public void ParameterProvenanceMustBeAnArrayAtEveryPublicBoundary(string form)
    {
        var (schemas, plan) = Documents(); var original = Undo(schemas, plan);
        foreach (var document in new[] { original, schemas })
        {
            var record = Records(document).First();
            if (form == "absent") record.Remove("parameterSources");
            else record["parameterSources"] = form == "null" ? null : JsonValue.Create("invalid");
            Assert.Contains(Resolution.ValidateAccounting(document), error => error.Contains("parameterSources array"));
        }
        string before = original.ToJsonString();
        Assert.Throws<InvalidOperationException>(() => Resolution.Apply(original, plan));
        Assert.Equal(before, original.ToJsonString());
        Assert.Contains(Resolution.ValidateApplied(schemas, plan), error => error.Contains("parameterSources array"));
    }

    private static (JsonObject Schema, JsonObject Plan) Documents() => (Read("node-schemas.json"), Read(Resolution.PlanFileName));
    private static JsonObject Read(string name)
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            string path = Path.Combine(directory.FullName, "docs", "capabilities", name);
            if (File.Exists(path)) return JsonNode.Parse(File.ReadAllText(path))!.AsObject();
        }
        throw new InvalidOperationException("Repository capability fixture is unavailable.");
    }
    private static string[] Strings(JsonNode node) => node.AsArray().Select(v => v!.GetValue<string>()).ToArray();
    private static IEnumerable<JsonObject> Records(JsonObject schemas) => schemas["records"]!.AsArray().OfType<JsonObject>();
    private static string Key(JsonObject record) => string.Join(":", new[] { "module", "sourceClass", "nodeId" }.Select(k => record[k]!.GetValue<string>()));
    private static JsonNode? At(JsonObject record, string path)
    {
        JsonNode? value = record;
        foreach (string part in path.Split('/')) value = value is JsonArray array ? array[int.Parse(part)] : value![part];
        return value;
    }
    private static void Set(JsonObject record, string path, JsonNode value)
    {
        int last = path.LastIndexOf('/'); var parent = At(record, path[..last]); string key = path[(last + 1)..];
        if (parent is JsonArray array) array[int.Parse(key)] = value; else parent![key] = value;
    }
    // Restore only the reviewed symbolic occurrences. No source code is imported or evaluated.
    private static JsonObject Undo(JsonObject published, JsonObject plan)
    {
        var original = published.DeepClone().AsObject(); original.Remove("staticResolution");
        foreach (var occurrence in plan["occurrences"]!.AsArray().OfType<JsonObject>())
        {
            var record = Records(original).Single(r => Key(r) == Key(occurrence));
            Set(record, occurrence["path"]!.GetValue<string>(), new JsonObject
            {
                ["resolution"] = "unresolved", ["expression"] = occurrence["expression"]!.DeepClone(),
                ["source"] = occurrence["source"]!.DeepClone(), ["reason"] = occurrence["reason"]!.DeepClone()
            });
        }
        var summaries = new JsonArray(); int total = 0, partial = 0;
        foreach (var record in Records(original))
        {
            record.Remove("resolvedFields"); var mirrors = new JsonArray();
            foreach (string field in new[] { "inputs", "outputs", "metadata" }) Walk(record[field], field, mirrors);
            record["unresolvedFields"] = mirrors; total += mirrors.Count;
            record["schemaStatus"] = mirrors.Count == 0 ? "resolved-static" : "partial-static";
            if (mirrors.Count > 0)
            {
                partial++;
                summaries.Add(new JsonObject { ["nodeId"] = record["nodeId"]!.DeepClone(), ["module"] = record["module"]!.DeepClone(),
                    ["fields"] = mirrors.Count, ["status"] = "partial-static" });
            }
            foreach (var parameter in record["parameterSources"]!.AsArray().OfType<JsonObject>())
            {
                var values = new JsonArray(); Walk(At(record, parameter["path"]!.GetValue<string>()), "", values);
                parameter["resolution"] = values.Count == 0 ? "resolved-static" : "partial-static";
            }
        }
        original["coverage"] = new JsonObject { ["localNodeRecords"] = Records(original).Count(),
            ["statuses"] = new JsonObject { ["partial-static"] = partial, ["resolved-static"] = Records(original).Count() - partial },
            ["unresolvedExpressionCount"] = total };
        original["unresolvedCases"] = summaries;
        return original;
    }
    private static void Walk(JsonNode? node, string path, JsonArray result)
    {
        if (node is JsonObject obj)
        {
            if (obj["resolution"] is JsonValue resolution && resolution.TryGetValue<string>(out var value) && value == "unresolved")
            { var mirror = obj.DeepClone().AsObject(); mirror["path"] = path; result.Add(mirror); }
            foreach (var pair in obj) Walk(pair.Value, $"{path}/{pair.Key}", result);
        }
        else if (node is JsonArray array)
            for (int i = 0; i < array.Count; i++) Walk(array[i], $"{path}/{i}", result);
    }
}
