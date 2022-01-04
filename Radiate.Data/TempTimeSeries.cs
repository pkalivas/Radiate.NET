using System.IO;
using Newtonsoft.Json;
using Radiate.Data.Models;

namespace Radiate.Data;

public class TempTimeSeries : IDataSet
{
    private readonly int _featureLimit;

    public TempTimeSeries(int featureLimit)
    {
        _featureLimit = featureLimit;
    }
    
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var fileLocation = $"{Environment.CurrentDirectory}\\DataSets\\TempTimeSeries\\TempTimeSeries.json";
        var contents = await File.ReadAllTextAsync(fileLocation);
        var data = JsonConvert.DeserializeObject<List<TempOverTime>>(contents)
            .Take(_featureLimit)
            .ToList();

        var result = data.Select(val => new[] { val.Temp }).ToList();

        return (result, result);
    }
}