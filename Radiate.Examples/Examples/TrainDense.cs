using Radiate.Data;
using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples;

public class TrainDense : IExample
{
    public async Task Run()
    {
        const int maxEpoch = 500;
        
        var (inputs, targets) = await new XOR().GetDataSet();

        var pair = new FeatureTargetPair(inputs, targets);

        var trainInputs = pair.TrainingInputs;
        
        var mlp = new MultiLayerPerceptron(new GradientInfo { Gradient = Gradient.SGD })
            .AddLayer(new DenseInfo(32, Activation.ReLU))
            .AddLayer(new DenseInfo(1, Activation.Sigmoid));

        var optimizer = new Optimizer(mlp, Loss.MSE);
        await optimizer.Train(trainInputs, epoch => epoch.Index == maxEpoch);
        
        foreach (var (ins, outs) in trainInputs)
        {
            var pred = optimizer.Predict(ins.First());
            Console.WriteLine($"Answer {outs[0]} Confidence {pred.Confidence}");
        }
    }
}