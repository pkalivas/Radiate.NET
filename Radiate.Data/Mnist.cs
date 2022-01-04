using Radiate.Data.Models;
using Radiate.Data.Utils;

namespace Radiate.Data;

public class Mnist : IDataSet
{
    private readonly int _featureLimit;

    public Mnist(int featureLimit)
    {
        _featureLimit = featureLimit;
    }
    
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var testFeaturesLocation = $"{Environment.CurrentDirectory}\\DataSets\\Minst\\test.gz";
        var features = (await Utilities.UnzipGZAndLoad<List<MinstImage>>(testFeaturesLocation))
            .Take(_featureLimit)
            .ToList();
        
        var rawInputs = features
            .Select(diget => diget.Image.Select(point => (float)point).ToArray())
            .ToList();
        var rawLabels = features
            .Select(diget => new List<float> { diget.Label }.ToArray())
            .ToList();

        return (rawInputs, rawLabels);
    }
}