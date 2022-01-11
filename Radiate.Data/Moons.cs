namespace Radiate.Data;

public class Moons : IDataSet
{
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var fileName = Path.Combine(Environment.CurrentDirectory, "DataSets", "Moons", "moons.csv");
        return await Utils.Utilities.LoadCsv(fileName);
    }
}