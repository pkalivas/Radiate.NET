using System.Reflection;

namespace Radiate.Data;

public class BostonHousing : IDataSet
{
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var assembly = Assembly.GetExecutingAssembly();
        var contents = await new StreamReader(assembly.GetManifestResourceStream("Radiate.Data.DataSets.BostonHousing.Boston.csv")).ReadToEndAsync();
        return Utils.Utilities.ReadCsv(contents);
    }
}