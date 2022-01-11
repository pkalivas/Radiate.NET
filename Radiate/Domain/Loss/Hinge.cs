using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Loss;

public class Hinge : ILossFunction
{
    public Cost Calculate(Tensor output, Tensor target)
    {
        var result = Tensor.Like(output.Shape);

        for (var i = 0; i < result.Count(); i++)
        {
            result[i] = Math.Max(0, 1f - target[i] * output[i]);
        }

        return new Cost(result, result.Sum());
    }
}