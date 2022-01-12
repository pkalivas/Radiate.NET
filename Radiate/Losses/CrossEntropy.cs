using Radiate.Extensions;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Losses;

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