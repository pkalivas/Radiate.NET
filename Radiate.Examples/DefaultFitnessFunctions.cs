using Radiate.Optimizers;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Tensors;

namespace Radiate.Examples;

public static class DefaultFitnessFunctions
{
    public static float MeanSquaredError<T>(T member, IEnumerable<(Tensor features, Tensor targets)> data) where T : IGenome, IPredictionModel
    {
        var total = 0.0f;
        foreach (var (features, targets) in data)
        {
            var output = member.Predict(features);
            total += (float) Math.Pow((output.Confidence - targets.Max()), 2);
        }
        
        return 1f - (total / data.Count());
    }
}