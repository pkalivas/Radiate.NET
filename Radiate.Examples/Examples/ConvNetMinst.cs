using Radiate.Data;
using Radiate.Domain.Activation;
using Radiate.Domain.Callbacks;
using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Extensions;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Examples.Callbacks;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples;

public class ConvNetMinst : IExample
{
    public async Task Run()
    {
        const int featureLimit = 5000;
        const int batchSize = 128;
        const int maxEpochs = 25;
        
        var (rawInputs, rawLabels) = await new Mnist(featureLimit).GetDataSet();
        
        var normalizedInputs = rawInputs.Normalize();
        var oneHotEncode = rawLabels.OneHotEncode();
        
        var pair = new TensorTrainSet(normalizedInputs, oneHotEncode)
            .Transform(new Shape(28, 28, 1))
            .Batch(batchSize)
            .Split();
        
        var neuralNetwork = new MultiLayerPerceptron()
            .AddLayer(new ConvInfo(16, 3))
            .AddLayer(new MaxPoolInfo(2))
            .AddLayer(new FlattenInfo())
            .AddLayer(new DenseInfo(64, Activation.Sigmoid))
            .AddLayer(new DenseInfo(pair.OutputSize, Activation.SoftMax));
        
        var optimizer = new Optimizer<MultiLayerPerceptron>(neuralNetwork, pair, new List<ITrainingCallback>
        {
            new VerboseTrainingCallback(pair, maxEpochs),
            new ModelWriterCallback()
        });
        
        await optimizer.Train(epoch => maxEpochs == epoch.Index);
    }
}