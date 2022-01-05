namespace Radiate.Data;

public class BreastCancer : IDataSet
{
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var fileName = Path.Combine(Environment.CurrentDirectory, "DataSets", "BreastCancer", "BreastCancer.csv");
        return await Utils.Utilities.LoadCsv(fileName);

    }
}