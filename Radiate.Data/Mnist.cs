using Radiate.Data.Models;
using Radiate.Data.Utils;
using Radiate.Domain.Services;

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
            .Select(diget => diget.Image.Select(point => ((float)point / 255) - 0.5f).ToList())
            .ToList();
        var rawLabels = features
            .Select(diget => diget.Label)
            .ToList();

        var normalizedInputs = FeatureService.Normalize(rawInputs);
        var oneHotEncode = FeatureService.OneHotEncode(rawLabels);
        
        return (normalizedInputs, oneHotEncode);
    }
}