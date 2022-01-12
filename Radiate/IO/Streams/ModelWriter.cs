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

    public static Optimizer<T> FromJson<T>(string optimizer) where T : class
    {
        var model = JsonConvert.DeserializeObject<OptimizerWrap>(optimizer);
        return Optimizer<T>.Load(model);
    }

    public static async Task<Optimizer<T>> FromStream<T>(Stream stream) where T : class
    {
        var reader = new StreamReader(stream);
        var model = await reader.ReadToEndAsync();
        return FromJson<T>(model);
    }
}