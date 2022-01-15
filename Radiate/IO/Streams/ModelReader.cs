using System.IO;
using Newtonsoft.Json;
using Radiate.IO.Wraps;
using Radiate.Optimizers;

namespace Radiate.IO.Streams;

public static class ModelReader
{
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