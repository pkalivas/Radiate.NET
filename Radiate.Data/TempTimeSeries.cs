using System.Reflection;
using Newtonsoft.Json;
using Radiate.Data.Models;

namespace Radiate.Data;

public class TempTimeSeries : IDataSet
{
    private readonly int _featureLimit;

    public TempTimeSeries(int featureLimit = 0)
    {
        _featureLimit = featureLimit;
    }
    
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var assembly = Assembly.GetExecutingAssembly();
        var contents = await new StreamReader(assembly.GetManifestResourceStream("Radiate.Data.DataSets.TempTimeSeries.TempTimeSeries.json")).ReadToEndAsync();
        var data = JsonConvert.DeserializeObject<List<TempOverTime>>(contents)
            .ToList();

        if (_featureLimit > 0)
        {
            data = data.Take(_featureLimit).ToList();
        }

        var result = data.Select(val => new[] { val.Temp }).ToList();

        return (result, result);
    }
}