using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Supervised.Forest;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.SVM;
using Radiate.Optimizers.Unsupervised.Clustering;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Losses;

public delegate Cost LossFunction(Tensor output, Tensor target);

public static class LossFunctionResolver
{
    public static LossFunction Get(Loss loss) => loss switch
    {
        Loss.Difference => new Difference().Calculate,
        Loss.MSE => new MeanSquaredError().Calculate,
        Loss.CrossEntropy => new CrossEntropy().Calculate,
        Loss.Hinge => new Hinge().Calculate,
        _ => throw new Exception($"Loss {loss} is not implemented.")
    };

    public static LossFunction Get<T>(T model) => model switch
    {
        MultiLayerPerceptron or RandomForest or KMeans => new Difference().Calculate,
        SupportVectorMachine => new Hinge().Calculate,
        _ => new Difference().Calculate,
    };

}