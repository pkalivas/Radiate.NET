using Radiate.Data;
using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples;

public class TrainDense : IExample
{
    public async Task Run()
    {
        const int maxEpoch = 500;
        
        var (inputs, targets) = await new XOR().GetDataSet();

        var pair = new TensorPair(inputs, targets);

        var trainInputs = pair.TrainingInputs;
        
        var mlp = new MultiLayerPerceptron(new GradientInfo { Gradient = Gradient.SGD })
            .AddLayer(new DenseInfo(32, Activation.ReLU))
            .AddLayer(new DenseInfo(1, Activation.Sigmoid));

        var optimizer = new Optimizer<MultiLayerPerceptron>(mlp, Loss.MSE);
        var net = await optimizer.Train(trainInputs, epoch => epoch.Index == maxEpoch);
        
        foreach (var (ins, outs) in trainInputs)
        {
            var pred = net.Predict(ins.First());
            Console.WriteLine($"Answer {outs[0][0]} Confidence {pred.Confidence}");
        }
    }
}