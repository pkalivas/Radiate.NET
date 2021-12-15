using Radiate.Domain.Records;

namespace Radiate.Domain.Loss;

public class NoCostFunction : ILossFunction
{
    public Cost Calculate(float[] output, float[] target)
    {
        throw new System.NotImplementedException();
    }
}