namespace Radiate.Data;

public class ClusterBlob : IDataSet
{
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var fileName = $"{Environment.CurrentDirectory}\\DataSets\\BlobCluster\\blob.csv";
        var contents = await File.ReadAllTextAsync(fileName);

        var features = new List<float[]>();
        var labels = new List<float[]>();
        foreach (var row in contents.Split("\n").Skip(1))
        {
            var columns = row
                .Split(",")
                .Skip(1)
                .Select(Convert.ToSingle)
                .ToList();
            
            features.Add(columns.Take(columns.Count - 1).ToArray());
            labels.Add(columns.Skip(columns.Count - 1).ToArray());
        }
        
        return (features, labels);
    }
}