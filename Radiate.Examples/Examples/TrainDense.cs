using Radiate.Activations;
using Radiate.Callbacks;
using Radiate.Callbacks.Interfaces;
using Radiate.Data;
using Radiate.Gradients;
using Radiate.Losses;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;
using Radiate.Tensors;
using Radiate.Tensors.Enums;

namespace Radiate.Examples.Examples;

public class TrainDense : IExample
{
    public async Task Run()
    {
        const int maxEpoch = 100;
        
        var (inputs, targets) = await new Circles().GetDataSet();

        var pair = new TensorTrainSet(inputs, targets)
            .TransformTargets(Norm.OHE)
            .TransformFeatures(Norm.Standardize)
            .Shuffle()
            .Split().Compile();

        var o = new TensorTrainSet(pair.TensorOptions);
        var k = o.Process(inputs);

        var mlp = new MultiLayerPerceptron(new GradientInfo { Gradient = Gradient.SGD })
            .AddLayer(new DenseInfo(32, Activation.ReLU))
            .AddLayer(new DenseInfo(pair.OutputCategories, Activation.Sigmoid));

        var optimizer = new Optimizer<MultiLayerPerceptron>(mlp, pair, Loss.MSE, new List<ITrainingCallback>()
        {
            new ColorizedVerboseTrainingCallback(pair, maxEpoch, false),
            // new VerboseTrainingCallback(pair, maxEpoch, false),
            new ConfusionMatrixCallback()
        });
        
        await optimizer.Train(epoch => epoch.Index == maxEpoch);
    }
}