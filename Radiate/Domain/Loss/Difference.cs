using System.Linq;
using Radiate.Domain.Records;

namespace Radiate.Domain.Loss
{
    public class Difference : ILossFunction
    {
        public Cost Calculate(float[] output, float[] target)
        {
            var result = output.Zip(target)
                .Select(pair => pair.Second - pair.First)
                .ToArray();
            
            return new Cost(result, result.Sum());
        }
    }
}