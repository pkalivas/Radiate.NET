
namespace Radiate.Data;

public interface IDataSet
{
    Task<(List<float[]> inputs, List<float[]> targets)> GetDataSet();

}