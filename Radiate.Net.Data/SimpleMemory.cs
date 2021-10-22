using System.Collections.Generic;

namespace Radiate.Net.Data
{
    public class SimpleMemory : IDataSet
    {
        public (List<List<float>> inputs, List<List<float>> targets) GetDataSet()
        {
            var inputs = new List<List<float>>();
            inputs.Add(new() { 0f });
            inputs.Add(new() { 0f });
            inputs.Add(new() { 0f });
            inputs.Add(new() { 1f });
            inputs.Add(new() { 0f });
            inputs.Add(new() { 0f });
            inputs.Add(new() { 0f });

            var targets = new List<List<float>>();
            targets.Add(new() { 0f });
            targets.Add(new() { 0f });
            targets.Add(new() { 1f });
            targets.Add(new() { 0f });
            targets.Add(new() { 0f });
            targets.Add(new() { 0f });
            targets.Add(new() { 1f });

            return (inputs, targets);
        }
    }
}