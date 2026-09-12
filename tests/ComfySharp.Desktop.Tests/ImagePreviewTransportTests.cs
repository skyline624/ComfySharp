using System.Net;
using System.Text.Json.Nodes;
using ComfySharp.Desktop;
using Xunit;

namespace ComfySharp.Desktop.Tests;

public sealed class ImagePreviewTransportTests
{
    [Fact]
    public async Task Original_file_descriptor_is_encoded_in_the_local_route()
    {
        byte[] png = PngPreviewFixture.Create(); Uri? requested = null;
        using var client = new HttpClient(new Handler((request, _) =>
        {
            requested = request.RequestUri; var content = new ByteArrayContent(png); content.Headers.ContentType = new("image/png");
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = content });
        }));
        var result = await ImagePreviewTransport.ReadAsync(client, new Uri("http://127.0.0.1:8189"), new("a &é.png", "album &?/x", "temp"), TestContext.Current.CancellationToken);
        Assert.Equal(png, result); Assert.Equal("127.0.0.1", requested!.Host);
        Assert.Equal("/view?filename=a%20%26%C3%A9.png&subfolder=album%20%26%3F%2Fx&type=temp", requested.PathAndQuery);
    }

    [Theory]
    [InlineData("http://example.com")]
    [InlineData("https://localhost")]
    [InlineData("http://user@localhost")]
    public async Task Nonlocal_or_unsupported_host_addresses_never_send_requests(string address)
    {
        using var client = new HttpClient(new Handler((_, _) => throw new InvalidOperationException("Must not send")));
        await Assert.ThrowsAsync<InvalidDataException>(() => ImagePreviewTransport.ReadAsync(client, new Uri(address), new("x.png", "", "output"), TestContext.Current.CancellationToken));
    }

    [Theory]
    [InlineData("wrong-type")]
    [InlineData("oversized")]
    [InlineData("missing")]
    [InlineData("redirect")]
    public async Task Invalid_responses_fail_before_decoding(string scenario)
    {
        using var client = new HttpClient(new Handler((_, _) =>
        {
            var content = new ByteArrayContent([1, 2]); content.Headers.ContentType = new(scenario == "wrong-type" ? "text/html" : "image/png");
            if (scenario == "oversized") content.Headers.ContentLength = ImagePreviewTransport.MaximumPngBytes + 1L;
            var response = new HttpResponseMessage(scenario == "missing" ? HttpStatusCode.NotFound : scenario == "redirect" ? HttpStatusCode.Found : HttpStatusCode.OK) { Content = content };
            response.Headers.Location = new Uri("https://example.com/image.png"); return Task.FromResult(response);
        }));
        if (scenario is "missing" or "redirect") await Assert.ThrowsAsync<HttpRequestException>(() => ImagePreviewTransport.ReadAsync(client, new Uri("http://localhost"), new("x.png", "", "output"), TestContext.Current.CancellationToken));
        else await Assert.ThrowsAsync<InvalidDataException>(() => ImagePreviewTransport.ReadAsync(client, new Uri("http://localhost"), new("x.png", "", "output"), TestContext.Current.CancellationToken));
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("{\"filename\":\"x.png\",\"subfolder\":\"\",\"type\":\"input\"}")]
    [InlineData("{\"filename\":\"../x.png\",\"subfolder\":\"\",\"type\":\"output\"}")]
    [InlineData("{\"filename\":\"x.jpg\",\"subfolder\":\"\",\"type\":\"temp\"}")]
    public void Unsupported_descriptors_are_rejected(string json) => Assert.Throws<InvalidDataException>(() => PreviewImageFile.Parse(JsonNode.Parse(json)));

    [Fact]
    public void Submission_snapshots_prompt_workflow_unknown_fields_and_targets()
    {
        var prompt = JsonNode.Parse("""{"1":{"class_type":"SaveImage","inputs":{"filename_prefix":"mine"}}}""")!.AsObject();
        var workflow = JsonNode.Parse("""{"version":1,"nodes":[],"unknown":{"keep":["猫",null,2]}}""")!.AsObject();
        string originalPrompt = prompt.ToJsonString(), originalWorkflow = workflow.ToJsonString(); string[] targets = ["1"];
        var body = HostSupervisor.CreateSubmission(prompt, "desktop", targets, workflow);
        prompt.Clear(); workflow.Clear(); targets[0] = "changed";
        Assert.Equal(originalPrompt, body["prompt"]!.ToJsonString());
        Assert.Equal(originalWorkflow, body["extra_data"]!["extra_pnginfo"]!["workflow"]!.ToJsonString());
        Assert.Equal("1", body["partial_execution_targets"]![0]!.GetValue<string>());
        Assert.Equal("desktop", body["client_id"]!.GetValue<string>());
        Assert.False(HostSupervisor.CreateSubmission(new(), "desktop").ContainsKey("extra_data"));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Stream_is_bounded_without_content_length_and_disposed_on_cancellation(bool cancel)
    {
        var stream = new GeneratedStream(cancel); using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        using var client = new HttpClient(new Handler((_, _) =>
        {
            var content = new StreamContent(stream); content.Headers.ContentType = new("image/png");
            Assert.Null(content.Headers.ContentLength); return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = content });
        }));
        var reading = ImagePreviewTransport.ReadAsync(client, new Uri("http://localhost"), new("x.png", "", "output"), cancellation.Token);
        if (cancel)
        {
            await stream.Started.Task; cancellation.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => reading);
        }
        else
        {
            await Assert.ThrowsAsync<InvalidDataException>(() => reading);
            Assert.Equal(ImagePreviewTransport.MaximumPngBytes + 1L, stream.ReadBytes);
        }
        Assert.True(stream.IsDisposed);
    }

    private sealed class GeneratedStream(bool block) : Stream
    {
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public bool IsDisposed { get; private set; }
        public long ReadBytes { get; private set; }
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            Started.TrySetResult();
            if (block) await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            int count = (int)Math.Min(buffer.Length, ImagePreviewTransport.MaximumPngBytes + 1L - ReadBytes);
            buffer.Span[..count].Clear(); ReadBytes += count; return count;
        }
        protected override void Dispose(bool disposing) { IsDisposed = true; base.Dispose(disposing); }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override void Flush() => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private sealed class Handler(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> send) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) => send(request, cancellationToken);
    }
}
