using System.Collections.Generic;

namespace Radiate.Net.Data.Utils
{
    public static class Utilities
    {
        public static (List<List<float>> inputs, List<List<float>> targets) Layer(List<List<float>> ins, List<List<float>> outs, int size)
        {
            var inputs = new List<List<float>>();
            var targets = new List<List<float>>();

            for (var i = size; i < ins.Count; i++)
            {
                var tempIn = new List<float>();
                var tempOut = new List<float>();
                for (var j = 0; j < size; j++)
                {
                    tempIn.AddRange(ins[i - j]);
                }
                
                inputs.Add(tempIn);
                targets.Add(outs[i]);
            }

            return (inputs, targets);
        }
    }
}