using System.IO.Compression;
using System.Text.Json;
using Radiate.Data.Models;

namespace Radiate.Data;
// var inputShape = new Shape(32, 32, 3);

public class Cifar : IDataSet
{
    private readonly int _featureLimit;

    public Cifar(int featureLimit)
    {
        _featureLimit = featureLimit;
    }
    
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var dataLocation = Path.Combine(Environment.CurrentDirectory, "DataSets", "cifar", "images.zip");

        using var zip = new ZipArchive(File.OpenRead(dataLocation), ZipArchiveMode.Read);

        var features = new List<CifarImage>();
        for (var i = 0; i < _featureLimit; i++)
        {
            var zipFile = zip.Entries[i].Open();
            features.Add(JsonSerializer.Deserialize<CifarImage>(zipFile));
        }
        
        var rawInputs = features
            .Select(diget => diget.Image.Select(point => (float)point).ToArray())
            .ToList();
        var rawLabels = features
            .Select(diget => new List<float> { diget.Label }.ToArray())
            .ToList();

        return (rawInputs, rawLabels);
    }
}