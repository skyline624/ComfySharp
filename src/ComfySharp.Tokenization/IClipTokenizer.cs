namespace ComfySharp.Tokenization;

/// <summary>The text-only CLIP contract, independent of tensor runtimes.</summary>
public interface IClipTokenizer
{
    int BosTokenId { get; }
    int EosTokenId { get; }
    IReadOnlyList<int> Encode(string text, bool addSpecialTokens = true, CancellationToken cancellationToken = default);
    string Decode(IEnumerable<int> ids, bool skipSpecialTokens = true);
    string GetToken(int id);
}
