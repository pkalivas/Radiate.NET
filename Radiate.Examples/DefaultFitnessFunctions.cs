using Radiate.Optimizers;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Tensors;

namespace Radiate.Examples;

public static class DefaultFitnessFunctions
{
    public static float MeanSquaredError<T>(T member, IEnumerable<Tensor> features, IEnumerable<Tensor> targets) where T : IGenome, IPredictionModel
    {
        var total = 0.0f;
        foreach (var points in features.Zip(targets))
        {
            var output = member.Predict(points.First);
            total += (float) Math.Pow((output.Confidence - points.Second.Max()), 2);
        }
        
        return 1f - (total / features.Count());
    }
}