using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Loss;

public class Difference : ILossFunction
{
    public Cost Calculate(Tensor output, Tensor target)
    {
        var result = output.Read1D().Zip(target.Read1D())
            .Select(pair => pair.Second - pair.First)
            .ToTensor();
        
        return new Cost(result, result.Sum());
    }
}