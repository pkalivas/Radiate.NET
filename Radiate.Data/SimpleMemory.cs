using System.Collections.Generic;

namespace Radiate.Data
{
    public class SimpleMemory : IDataSet
    {
        public (List<float[]> inputs, List<float[]> targets) GetDataSet()
        {
            var inputs = new List<float[]>
            {
                new float[1] { 0 },
                new float[1] { 0 },
                new float[1] { 0 },
                new float[1] { 1 },
                new float[1] { 0 },
                new float[1] { 0 },
                new float[1] { 0 }
            };

            var targets = new List<float[]>
            {
                new float[1] { 0 },
                new float[1] { 0 },
                new float[1] { 1 },
                new float[1] { 0 },
                new float[1] { 0 },
                new float[1] { 0 },
                new float[1] { 1 }
            };

            return (inputs, targets);
        }
    }
}