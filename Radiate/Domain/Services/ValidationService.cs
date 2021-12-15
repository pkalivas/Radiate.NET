
namespace Radiate.Domain.Services;

public static class ValidationService
{
    public static float ClassificationAccuracy(List<float[]> predictions, List<float[]> targets)
    {
        var correctClasses = predictions.Zip(targets)
            .Select(pair =>
            {
                var firstMax = pair.First.ToList().IndexOf(pair.First.Max());
                var secondMax = pair.Second.ToList().IndexOf(pair.Second.Max());

                return firstMax == secondMax ? 1f : 0f;
            })
            .Sum();

        return correctClasses / predictions.Count;
    }

    public static float RegressionAccuracy(List<float[]> predictions, List<float[]> targets)
    {
        var targetTotal = targets.Sum(tar => tar.First());
        var absoluteDifference = predictions.Zip(targets)
            .Select(pair => Math.Abs(pair.Second.First() - pair.First.First()))
            .Sum();

        return (targetTotal - absoluteDifference) / targetTotal;
    }
}