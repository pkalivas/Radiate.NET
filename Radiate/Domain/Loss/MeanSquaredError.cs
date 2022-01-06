using Radiate.Domain.Extensions;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Loss;

public class MeanSquaredError : ILossFunction
{
    public Cost Calculate(Tensor output, Tensor target)
    {
        var errors = new List<float>();
        var squaredErrors = new List<float>();

        foreach (var (guess, label) in output.Zip(target))
        {
            var difference = label - guess;
            
            errors.Add(difference);
            squaredErrors.Add((float) Math.Pow(difference, 2));
        }

        var loss = squaredErrors.Sum() / output.Count();
        
        return new Cost(squaredErrors.ToTensor(), loss);
    }
}