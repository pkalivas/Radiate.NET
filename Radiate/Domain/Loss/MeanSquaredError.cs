using Radiate.Domain.Records;

namespace Radiate.Domain.Loss;

public class MeanSquaredError : ILossFunction
{
    public Cost Calculate(float[] output, float[] target)
    {
        var errors = new List<float>();
        var squaredErrors = new List<float>();

        foreach (var (guess, label) in output.Zip(target))
        {
            var difference = label - guess;
            
            errors.Add(difference);
            squaredErrors.Add((float) Math.Pow(difference, 2));
        }

        var loss = squaredErrors.Sum() / output.Length;
        
        return new Cost(errors.ToArray(), loss);
    }
}