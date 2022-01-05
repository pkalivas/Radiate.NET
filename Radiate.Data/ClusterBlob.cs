namespace Radiate.Data;

public class ClusterBlob : IDataSet
{
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var fileName = Path.Combine(Environment.CurrentDirectory, "DataSets", "BlobCluster", "blob.csv");
        return await Utils.Utilities.LoadCsv(fileName);

    }
}