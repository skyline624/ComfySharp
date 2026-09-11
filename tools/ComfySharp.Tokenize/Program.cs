using ComfySharp.Tokenize;

using var cancellation = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancellation.Cancel(); };
return await TokenizeCommand.RunAsync(args, Console.In, Console.Out, Console.Error, cancellation.Token);
