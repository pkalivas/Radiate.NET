
namespace Radiate.Domain.Loss;

public static class LossFunctionFactory
{
    public static ILossFunction Get(Loss loss) => loss switch
    {
        Loss.Difference => new Difference(),
        Loss.MSE => new MeanSquaredError(),
        Loss.CrossEntropy => new CrossEntropy(),
        _ => throw new Exception($"Loss {loss} is not implemented.")
    };
}