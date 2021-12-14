using System.Collections.Generic;

namespace Radiate.Data
{
    public interface IDataSet
    {
        (List<float[]> inputs, List<float[]> targets) GetDataSet();

    }
}