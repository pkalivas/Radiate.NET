using Radiate.NET.Domain.Records;

namespace Radiate.NET.Domain.Loss
{
    public interface ILossFunction
    {
        Cost Calculate(float[] output, float[] target);
    }
}