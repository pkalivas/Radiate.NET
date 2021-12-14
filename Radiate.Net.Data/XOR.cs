using System.Collections.Generic;

namespace Radiate.Net.Data
{
    public class XOR : IDataSet
    {
        public (List<float[]> inputs, List<float[]> targets) GetDataSet()
        {
            var inputs = new List<float[]>
            {
                new float[2] { 0, 0 },
                new float[2] { 1, 1 },
                new float[2] { 1, 0 },
                new float[2] { 0, 1 }
            };

            var answers = new List<float[]>
            {
                new float[1] { 0 },
                new float[1] { 0 },
                new float[1] { 1 },
                new float[1] { 1 }
            };

            return (inputs, answers);
        }
    }
}