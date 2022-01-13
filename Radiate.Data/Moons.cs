using System.Reflection;

namespace Radiate.Data;

public class Moons : IDataSet
{
    public async Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet()
    {
        var assembly = Assembly.GetExecutingAssembly();
        var contents = await new StreamReader(assembly.GetManifestResourceStream("Radiate.Data.DataSets.Moons.moons.csv")).ReadToEndAsync();
        return Utils.Utilities.ReadCsv(contents);
        
        // var fileName = Path.Combine(Environment.CurrentDirectory, "DataSets", "Moons", "moons.csv");
        // return await Utils.Utilities.LoadCsv(fileName);
    }
}