using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Loss;

public class EuclideanDistance : ILossFunction
{
    public Cost Calculate(Tensor output, Tensor target)
    {
        var result = Tensor.Like(output.Shape);
        var total = 0f;
        for (var i = 0; i < output.Shape.Height; i++)
        {
            var difference = output[i] - target[i];

            total += difference;
            result[i] = (float) Math.Sqrt(difference * difference);
        }

        var error = (float)Math.Sqrt(Math.Pow(total, 2));
        return new Cost(result, error);
    }
}