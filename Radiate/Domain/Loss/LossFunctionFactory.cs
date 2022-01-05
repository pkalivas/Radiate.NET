
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Loss;

public delegate Cost LossFunction(Tensor output, Tensor target);

public static class LossFunctionResolver
{
    public static LossFunction Get(Loss loss) => loss switch
    {
        Loss.Difference => new Difference().Calculate,
        Loss.MSE => new MeanSquaredError().Calculate,
        Loss.CrossEntropy => new CrossEntropy().Calculate,
        _ => throw new Exception($"Loss {loss} is not implemented.")
    };
}