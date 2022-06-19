using Radiate.Extensions;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Losses;

public class Difference : ILossFunction
{
    public Loss LossType() => Loss.Difference;

    public Cost Calculate(Tensor output, Tensor target)
    {
        var result = output.Zip(target)
            .Select(pair => pair.Second - pair.First)
            .ToTensor();
        
        return new Cost(result, result.Sum());
    }
}