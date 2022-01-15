using System.IO;
using Newtonsoft.Json;
using Radiate.IO.Wraps;
using Radiate.Optimizers;

namespace Radiate.IO.Streams;

public static class ModelWriter
{
    public static string ToJson<T>(Optimizer<T> optimizer) where T : class
    {
        var wrap = optimizer.Save();
        return JsonConvert.SerializeObject(wrap);
    }

    public static async Task<Stream> ToStream<T>(Optimizer<T> optimizer) where T : class
    {
        var wrap = ToJson(optimizer);
        var memoryStream = new MemoryStream();
        var writer = new StreamWriter(memoryStream);

        await writer.WriteAsync(wrap);
        await writer.FlushAsync();
        memoryStream.Position = 0;

        return memoryStream;
    }
}