using System.IO.Compression;
using System.Reflection;
using System.Text.Json;
using Radiate.Data.Models;

namespace Radiate.Data;
// var inputShape = new Shape(32, 32, 3);

public class Cifar : IDataSet
{
    private readonly int _featureLimit;

    public Cifar(int featureLimit = 0)
    {
        _featureLimit = featureLimit;
    }
    
    public Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var assembly = Assembly.GetExecutingAssembly();
        var contents = new StreamReader(assembly.GetManifestResourceStream("Radiate.Data.DataSets.cifar.images.zip"));

        using var zip = new ZipArchive(contents.BaseStream, ZipArchiveMode.Read);

        var features = new List<CifarImage>();
        var entries = zip.Entries;
        var iter = _featureLimit == 0 ? entries.Count : _featureLimit;
        for (var i = 0; i < iter; i++)
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

        return Task.FromResult((rawInputs, rawLabels));
    }
}