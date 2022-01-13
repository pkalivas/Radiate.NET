using System.Reflection;
using Radiate.Data.Models;
using Radiate.Data.Utils;

namespace Radiate.Data;

public class Mnist : IDataSet
{
    private readonly int _featureLimit;

    public Mnist(int featureLimit = 0)
    {
        _featureLimit = featureLimit;
    }
    
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var assembly = Assembly.GetExecutingAssembly();
        var contents = new StreamReader(assembly.GetManifestResourceStream("Radiate.Data.DataSets.Minst.test.gz"));

        var features = (await Utilities.UnzipGZAndLoad<List<MinstImage>>(contents.BaseStream));

        if (_featureLimit > 0)
        {
            features = features.Take(_featureLimit).ToList();
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