using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Loss;

public class MeanSquaredError : ILossFunction
{
    public Cost Calculate(Tensor output, Tensor target)
    {
        var errors = new List<float>();
        var squaredErrors = new List<float>();

        foreach (var (guess, label) in output.Read1D().Zip(target.Read1D()))
        {
            var difference = label - guess;
            
            errors.Add(difference);
            squaredErrors.Add((float) Math.Pow(difference, 2));
        }

        var loss = squaredErrors.Sum() / output.Read1D().Length;
        
        return new Cost(errors.ToTensor(), loss);
    }
}