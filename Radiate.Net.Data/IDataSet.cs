using System.Collections.Generic;

namespace Radiate.Net.Data
{
    public interface IDataSet
    {
        (List<float[]> inputs, List<float[]> targets) GetDataSet();

    }
}