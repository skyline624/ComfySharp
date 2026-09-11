# CLIP text tokenization: dependency profile and provenance

This tranche covers CLIP text normalization, byte-level BPE, ComfyUI prompt weights and SD1/SDXL sequence packing. The functional reference remains [ComfyUI at `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a). Token IDs and weights do not qualify text encoders, model inference, masks, pooled outputs or image generation. Textual inversion resolution and other tokenizer families remain separate capabilities.

## Exact dependency profile

The reference laboratory was reconfirmed on 11 September 2026: Python 3.12.10, **Transformers 5.14.1 / tokenizers 0.22.2**. It loaded only the four local tokenizer resources with `local_files_only=True`. No model weights were loaded. Python and the native tokenizers library are reference-laboratory tools, not product or test runtime dependencies.

ComfyUI's [frozen requirements](https://github.com/comfy-org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/requirements.txt) give dependency minima, not an exact normalization contract. This profile therefore explicitly selects the observed 5.14.1 / 0.22.2 behavior. Historical Transformers 4.50.3 uses another implementation with different preprocessing and optional `ftfy`; simultaneous compatibility with that profile is not claimed.

The installed `tokenization_clip.py` is byte-identical to the [official Transformers source at `a08ace4bbd97e721c98751deec37d87b026acadc`](https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/models/clip/tokenization_clip.py): 5,202 bytes, SHA-256 `fb5338a0d928deb9470bc84d4932fb2d6456611f1ce14ff11408cc3ec6f01ef7`. The effective backend was serialized independently before corpus production. Its configuration specifies:

- NFC, Unicode regex `\s+` replaced by one space, then scalar lowercase with expansions.
- CLIP splitting with `Removed`, `invert=true`, followed by ByteLevel with `add_prefix_space=false`, `trim_offsets=true`, `use_regex=true`.
- BPE suffix `</w>`, empty continuing prefix, no dropout, `fuse_unk=false`, unknown token EOS; 49,408 vocabulary IDs and 48,894 ordered merges.
- BOS 49406 and EOS/UNK/PAD 49407. Added BOS has `normalized=true`; added EOS has `normalized=false`; both are special, without single-word or strip flags.
- RobertaProcessing inserts outer BOS/EOS with `trim_offsets=false`, `add_prefix_space=false`. ByteLevel decode is followed by replacing `</w>` with a space and stripping in the CLIP wrapper.

The CLIP split pattern is:

```text
<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
```

The declared `model_max_length=8192` is not ComfyUI's default chunk length of 77 and must not cause BPE truncation. The declared `name_or_path` identifies provenance; it does not request remote loading.

## Canonical embedded resources

The four resources in `src/ComfySharp.Tokenization/Resources/Clip` come from the [frozen ComfyUI directory](https://github.com/comfy-org/ComfyUI/tree/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/sd1_tokenizer). They were exported using binary process capture of Git blobs, not text redirection or the Windows checkout. These hashes identify the embedded bytes, including canonical LF endings.

| Resource | Bytes | SHA-256 |
|---|---:|---|
| `vocab.json` | 1,059,962 | `e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349` |
| `merges.txt` | 524,619 | `9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a` |
| `tokenizer_config.json` | 808 | `f37d2056ee9743c44a79b85eb3763b7bab378339e69a4b50f58a37e89066821c` |
| `special_tokens_map.json` | 472 | `c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac` |

The source configuration names `openai/clip-vit-large-patch14`. Comparison with that [official Hugging Face repository at `32bd64288804d66eefd0ccbe215aa642df71cc41`](https://huggingface.co/openai/clip-vit-large-patch14/tree/32bd64288804d66eefd0ccbe215aa642df71cc41) establishes that the merge file is byte-identical and the vocabulary mapping is identical after parsing JSON. Its vocabulary serialization and configuration files differ from the frozen ComfyUI files; they were not substituted.

The asset lineage is also checked directly against [OpenAI CLIP at `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6). All 48,894 selected merges equal the list selected from `clip/bpe_simple_vocab_16e6.txt.gz` by OpenAI's `SimpleTokenizer`. Reconstructing the vocabulary from its exact `bytes_to_unicode` function, selected merges and two special tokens reproduces every one of the 49,408 canonical IDs. The original compressed resource SHA-256 is `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`; it is used only for provenance verification and is not embedded in the product. The OpenAI MIT notice is retained; this asset match does not imply that OpenAI's original text cleaning is the selected normalization profile.

The Hugging Face repository has no separate license file or license field in the inspected model card. The resource attribution therefore relies on the directly verified OpenAI resource lineage and its MIT notice, together with the frozen ComfyUI provenance for the copied declarations. It does not infer a license for unrelated model weights from a `name_or_path` string.

## Unicode reference dependencies

Unicode processing is part of the compatibility profile. A platform's current Unicode tables cannot silently replace the reference tables. The [official tokenizers 0.22.2 Python Cargo lock](https://github.com/huggingface/tokenizers/blob/f383101a26663708484cac0727792aad74f78234/bindings/python/Cargo.lock) pins `unicode-normalization-alignments 0.1.12`, `onig 6.5.1` and `onig_sys 69.9.1`.

The normalization crate's `src/tables.rs` explicitly declares Unicode **9.0.0**. Its published crate has SHA-256 `43f613e4fa046e69818dd287fdc4bc78175ff20331479dab6e1b0f98d57062de`, equal to the Cargo lock checksum. `onig_sys 69.9.1` has SHA-256 `c7f86c6eef3d6df15f23bcfb6af487cbd2fed4e5581d58d5bf1f5f8b7f6727dc`, also matching the lock; its bundled Oniguruma history records the Unicode **16.0** update. These published dependencies identify source evidence, not a claim that version metadata alone proves all installed behavior.

The [tokenizers lowercase implementation](https://github.com/huggingface/tokenizers/blob/f383101a26663708484cac0727792aad74f78234/tokenizers/src/tokenizer/normalizer.rs#L547) calls Rust `char::to_lowercase` for each scalar. The installed native binary has SHA-256 `7acb83f5b89136597e0d14b788d82917bf2870df94575bf77e12731c4e49c4df` and contains the Rust compiler source marker `ded5c06cf21d2b93bffd5d884aa6e96934ee4234`. The [official Rust tables at that commit](https://github.com/rust-lang/rust/blob/ded5c06cf21d2b93bffd5d884aa6e96934ee4234/library/core/src/unicode/unicode_data.rs) explicitly declare Unicode **17.0.0**. Thus normalization, regex categories and case conversion have distinct version provenance. The generated managed tables are qualified against the installed backend's complete scalar behavior; integration evidence records their hash and scope separately.

Prompt-weight numeric parsing and embedding-marker whitespace boundaries additionally follow the reference Python 3.12 Unicode **15.0.0** digit and `str.isspace` behavior. Their factual Nd digit offsets and whitespace set are independent of the tokenizers backend's regex, normalization and lowercase tables. The port does not replace these separate language-level semantics with the host .NET Unicode version.

## Attribution and redistribution scope

ComfyUI prompt parsing and packing retain the project GPLv3 provenance. The adapted CLIP dependency behavior retains the source credit **Copyright 2021 The Open AI Team Authors and The HuggingFace Inc. team**, under Apache-2.0. OpenAI CLIP resources retain **Copyright (c) 2021 OpenAI**, under MIT. Unicode-related data/source notices are retained for Unicode, Rust, unicode-normalization-alignments and Oniguruma. See [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) and the full texts in `docs/licenses`.

This text-specific provenance record does not close the separate native dependency notice collection and SBOM gate for binary packaging.

## Managed implementation and input contract

`ComfySharp.Tokenization` has no package or tensor dependency. `ClipTokenizer` loads all required resources from its assembly, verifies their SHA-256 values and shares immutable vocabulary/merge/Unicode tables. Each instance has a bounded, synchronized word cache; words longer than 256 UTF-16 units are not cached. Encoding works with Unicode scalars and observes cancellation during normalization, splitting and BPE merging. Invalid UTF-16 is rejected explicitly. Decode accepts vocabulary IDs and reproduces the dependency's replacement behavior for incomplete or invalid UTF-8 byte sequences assembled from those IDs.

The embedded `Resources/ClipUnicode/unicode-profile.json` is 581,402 bytes, SHA-256 `ff294f10809b7012d1392387bd516859a2aa0d79d33a139ad6eb1d5d8002b842`. Its generator queried all **1,112,064 valid Unicode scalars** through the installed reference primitives. It records 677 letter ranges, 144 number ranges, 10 whitespace ranges, 1,488 lowercase mappings, 13,232 decompositions, 12,112 composition pairs and 814 nonzero combining classes. Combining classes were independently checked against the official Unicode 9 normalization crate table with no differences. Runtime normalization uses these pinned tables rather than the host OS's ICU or .NET Unicode version. Python 3.12's Unicode 15 decimal and whitespace facts used by prompt weighting are separate from these tokenizer tables.

`PromptWeights.Parse` implements the frozen parenthesis/escape semantics. Implicit nesting multiplies by 1.1; an explicit inner weight replaces its inherited weight. IEEE-754 doubles retain signed zero and nonfinite values. Brackets do not introduce weight syntax. `ComfyClipTokenizer` adds the source segmentation, word IDs, 7/8-token grouping rule, EOS and padding behavior for SD1-L and SDXL-L/G. `ComfySdxlTokenizer` emits G then L with independent length settings. Results are immutable and retain word IDs even when projected to token/weight pairs.

Construction and per-call options preserve the source length/weight precedence, including explicit null minimum-length overrides and left padding. Padding may legitimately exceed 77 tokens; tokenization alone does not establish that an encoder accepts that length. Configurations that cannot fit a small indivisible segment are rejected rather than reproducing an upstream infinite loop. With no textual-inversion resolver, `embedding:` is ordinary text under the source rules. Requesting resolution reports an unsupported capability; no embedding vector is synthesized.

The local diagnostic accepts one input through `--text`, `--file` (strict UTF-8, optional UTF-8 BOM) or `--stdin`, and a profile `sd1-l`, `sdxl-l`, `sdxl-g` or `sdxl`:

```sh
dotnet run --project tools/ComfySharp.Tokenize -- --text "a (cat:1.5)" --profile sdxl
```

It prints IDs, decoded text and weighted chunks as JSON. Nonfinite weights use explicit strings and every weight has a `weightBits` hexadecimal representation. It does not load weights, resolve textual inversion, initialize libtorch or contact a service. `--disable-weights` changes parsing only. Malformed UTF-8 and BOM-marked UTF-16/32 files are rejected instead of silently transcoded by BOM detection. BOM-less bytes that are valid UTF-8 are interpreted as UTF-8, including embedded controls.

## Independent comparison corpus

The separate reference laboratory executes the exact AST of the pinned ComfyUI functions/classes using canonical local assets. The .NET test suite reads JSON fixtures only; Python, Transformers and tokenizers are not required to run it. No expected output is generated by the C# implementation.

| Corpus | Source cases | Bytes | SHA-256 |
|---|---:|---:|---|
| `clip-text.reference.json` | 193: 113 BPE, 17 weighting, 59 packing, 4 wrapper composition | 2,225,425 | `bc92c74136a6ed38a314029f083305acbe8fae5aa8acb6c0808db3b8237f6cc2` |
| `clip-decode.reference.json` | 269: all 256 individual byte tokens, malformed/incomplete UTF-8 sequences and special-token cases | 35,086 | `bbaff41e4ebfc9fb76ba4da5cf60013cf41a9f05d6539a4738350ed7d76ac326` |

The text corpus was exported twice with identical bytes. An independent reviewer checked its source/resource/AST/configuration hashes and replayed every decoder case through the pinned dependency, with exact results. Assertions compare token IDs, vocabulary text, decoded strings, chunk lengths/order, word IDs and weight bits. NaNs require matching classification and sign; payload identity is not a portable arithmetic contract. No whitespace, numeric or token normalization is applied to expected outputs.

Two additional Windows differential probes compare the complete text pipeline to the dependency: an aggregate of affected scalars and all category-range boundaries (37,657 scalars, 52,944 tokens) and 2,006 random/targeted combining sequences (38,844 scalars, 90,810 tokens). Both token-ID arrays and decoded strings match exactly. These are samples of scalar interactions, not an exhaustive enumeration of possible strings. Input hashes are `f1a11fd102ed7833747a50d1f4c60d69f2389443c66a45bffce05a5d15cb31b6` and `31edc0a31f5d6a8c3b12a0b66cf69e59c103a9b56e76f4f63fa9423c7793ebcb` respectively. Integration/CI results are recorded separately in [QUALIFICATION.md](QUALIFICATION.md).

The [CI for source `cfe19ab`](https://github.com/skyline624/ComfySharp/actions/runs/34617953175) passes all **540 tokenization tests** on Windows, Ubuntu 24.04 and macOS 14 ARM64, with zero failures/skips, as part of 1,048 solution tests per OS. The fresh-process CLI also passes on all three platforms without Torch dependencies. The [structured evidence](qualification/clip-text-cfe19ab.json) records counts, revisions, hashes and limits. These tests qualify the stated text contract; no text encoder, model family, textual inversion or GPU workflow is thereby qualified.
