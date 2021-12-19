using System.Numerics;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Loss;

public class CrossEntropy : ILossFunction
{
    public Cost Calculate(Tensor output, Tensor target)
    {
        var errors = output.Read1D().Zip(target.Read1D())
            .Select(pair => -pair.Second * (float)Math.Log(pair.First))
            .ToTensor();
        
        return new Cost(errors, errors.Sum());
    }
}