
namespace Radiate.Domain.Services;

public static class ValidationService
{
    public static float ClassificationAccuracy(List<(float[] predictions, float[] targets)> outs)
    {
        var correctClasses = outs
            .Select(pair =>
            {
                var (first, second) = pair;
                var firstMax = first.ToList().IndexOf(first.Max());
                var secondMax = second.ToList().IndexOf(second.Max());

                return firstMax == secondMax ? 1f : 0f;
            })
            .Sum();

        return correctClasses / outs.Count;
    }

    public static float RegressionAccuracy(List<(float[] predictions, float[] targets)> outs)
    {
        var targetTotal = outs.Sum(tar => tar.targets.Sum());
        var absoluteDifference = outs
            .Select(pair => Math.Abs(pair.targets.First() - pair.predictions.First()))
            .Sum();

        return (targetTotal - absoluteDifference) / targetTotal;
    }
}