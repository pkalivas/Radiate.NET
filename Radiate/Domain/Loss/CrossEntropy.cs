using Radiate.Domain.Extensions;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Loss;

public class CrossEntropy : ILossFunction
{
    public Cost Calculate(Tensor output, Tensor target)
    {
        var errors = output.Zip(target)
            .Select(pair => -pair.Second * (float)Math.Log(pair.First))
            .ToTensor();
        
        return new Cost(errors, errors.Sum());
    }
}