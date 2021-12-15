using System.IO;
using Radiate.Domain.Services;

namespace Radiate.Data;

public class BostonHousing : IDataSet
{
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var fileName = $"{Environment.CurrentDirectory}\\DataSets\\BostonHousing\\Boston.csv";
        var contents = await File.ReadAllTextAsync(fileName);

        var features = new List<List<float>>();
        var labels = new List<float[]>();
        foreach (var row in contents.Split("\n").Skip(1))
        {
            var columns = row
                .Split(",")
                .Skip(1)
                .Select(Convert.ToSingle)
                .ToList();

            features.Add(columns.Take(columns.Count - 1).ToList());
            labels.Add(columns.Skip(columns.Count - 1).ToArray());
        }

        var normalizedFeatures = FeatureService.Standardize(features);

        return (normalizedFeatures, labels);
    }
}