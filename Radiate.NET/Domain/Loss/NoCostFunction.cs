using Radiate.NET.Domain.Records;

namespace Radiate.NET.Domain.Loss
{
    public class NoCostFunction : ILossFunction
    {
        public Cost Calculate(float[] output, float[] target)
        {
            throw new System.NotImplementedException();
        }
    }
}