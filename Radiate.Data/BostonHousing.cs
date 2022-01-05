namespace Radiate.Data;

public class BostonHousing : IDataSet
{
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var fileName = Path.Combine(Environment.CurrentDirectory, "DataSets", "BostonHousing", "Boston.csv");
        return await Utils.Utilities.LoadCsv(fileName);
    }
}