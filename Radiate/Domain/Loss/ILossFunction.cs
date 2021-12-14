using Radiate.Domain.Records;

namespace Radiate.Domain.Loss
{
    public interface ILossFunction
    {
        Cost Calculate(float[] output, float[] target);
    }
}