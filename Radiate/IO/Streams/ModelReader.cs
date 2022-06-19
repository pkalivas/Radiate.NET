using System.IO;
using Newtonsoft.Json;
using Radiate.IO.Wraps;
using Radiate.Optimizers;

namespace Radiate.IO.Streams;

public static class ModelReader
{
    public static Optimizer FromJson(string optimizer)
    {
        var model = JsonConvert.DeserializeObject<OptimizerWrap>(optimizer);
        return new Optimizer(model);
    }

    public static async Task<Optimizer> FromStream(Stream stream)
    {
        var reader = new StreamReader(stream);
        var model = await reader.ReadToEndAsync();
        return FromJson(model);
    }
}