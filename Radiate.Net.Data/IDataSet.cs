using System.Collections.Generic;

namespace Radiate.Net.Data
{
    public interface IDataSet
    {
        (List<List<float>> inputs, List<List<float>> targets) GetDataSet();

    }
}