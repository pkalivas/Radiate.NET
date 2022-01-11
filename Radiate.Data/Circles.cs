namespace Radiate.Data;

public class Circles : IDataSet
{
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var fileName = Path.Combine(Environment.CurrentDirectory, "DataSets", "Circles", "circles.csv");
        return await Utils.Utilities.LoadCsv(fileName);
    }
}