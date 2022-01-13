using System.Reflection;

namespace Radiate.Data;

public class IrisFlowers : IDataSet
{
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var assembly = Assembly.GetExecutingAssembly();
        var contents = await new StreamReader(assembly.GetManifestResourceStream("Radiate.Data.DataSets.Iris.iris.csv")).ReadToEndAsync();

        var features = new List<float[]>();
        var labels = new List<string>();
        foreach (var row in contents.Split("\n").Skip(1))
        {
            var columns = row
                .Split(",")
                .Skip(1)
                .ToList();
            
            features.Add(columns.Take(columns.Count - 1).Select(Convert.ToSingle).ToArray());
            labels.Add(columns.Skip(columns.Count - 1).First());
        }

        var uniqueLabels = labels.Distinct().ToList();
        var resultLabels = labels
            .Select(val => uniqueLabels.IndexOf(val))
            .Select(val => new float[] { Convert.ToSingle(val) })
            .ToList();
        
        return (features, resultLabels);
    }
}