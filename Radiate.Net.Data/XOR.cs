using System.Collections.Generic;

namespace Radiate.Net.Data
{
    public class XOR : IDataSet
    {
        public (List<List<float>> inputs, List<List<float>> targets) GetDataSet()
        {
            var inputs = new List<List<float>>();
            inputs.Add(new List<float> { 0, 0 });
            inputs.Add(new List<float> { 1, 1 });
            inputs.Add(new List<float> { 1, 0 });
            inputs.Add(new List<float> { 0, 1 });

            var answers = new List<List<float>>();
            answers.Add(new List<float> { 0 });
            answers.Add(new List<float> { 0 });
            answers.Add(new List<float> { 1 });
            answers.Add(new List<float> { 1 });

            return (inputs, answers);
        }
    }
}