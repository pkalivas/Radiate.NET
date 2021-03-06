using Radiate.Activations;
using Radiate.Callbacks;
using Radiate.Callbacks.Interfaces;
using Radiate.Data;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;
using Radiate.Records;
using Radiate.Tensors;
using Radiate.Tensors.Enums;

namespace Radiate.Examples.Examples;

public class ConvNetMinst : IExample
{
    public async Task Run()
    {
        const int featureLimit = 5000;
        const int batchSize = 32;
        const int maxEpochs = 5;
        
        var (rawInputs, rawLabels) = await new Mnist(featureLimit).GetDataSet();

        var pair = new TensorTrainSet(rawInputs, rawLabels)
            .Reshape(new Shape(28, 28, 1))
            .TransformFeatures(Norm.Image)
            .TransformTargets(Norm.OHE)
            .Batch(batchSize)
            .Split();
        
        var neuralNetwork = new MultiLayerPerceptron()
            .AddLayer(new ConvInfo(32, 3))
            .AddLayer(new MaxPoolInfo(2))
            .AddLayer(new FlattenInfo())
            .AddLayer(new DenseInfo(64, Activation.Sigmoid))
            .AddLayer(new DenseInfo(pair.OutputCategories, Activation.SoftMax));
        
        var optimizer = new Optimizer(neuralNetwork, pair, new List<ITrainingCallback>
        {
            new VerboseTrainingCallback(pair, maxEpochs),
            new ConfusionMatrixCallback()
        });
        
        await optimizer.Train<MultiLayerPerceptron>(epoch => maxEpochs == epoch.Index);
    }
}