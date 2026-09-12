# SaveLoRA export comparison laboratory

`verify_export.py` independently decodes the bounded safetensors byte layout of
an original adapter and a Host export. It checks unique JSON keys, dimensions,
dtype widths, contiguous data coverage and every tensor payload SHA-256 against
the Host report. It accepts the ten dtypes supported by the C# writer.

Run with a separate laboratory Python interpreter (standard library only):

```sh
python -I -B labs/save-lora-source/verify_export.py --original /shared/adapter.safetensors --export /evidence/output/loras/adapter_2_steps_00001_.safetensors --host-report /evidence/result.json --output /evidence/independent.json
```

The report must be a new file. Neither input is modified. This is a restricted
independent format check, not execution of the official safetensors package,
the training node or a model forward pass. Python is absent from the product
and distributed .NET tests. See [the node scope](../../docs/SAVE_LORA.md).
