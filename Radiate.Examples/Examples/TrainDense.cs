﻿using Radiate.Data;
using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Domain.Tensors;
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

        var pair = new TensorTrainSet(inputs, targets).Batch(1);

        var trainInputs = pair.TrainingInputs;
        
        var mlp = new MultiLayerPerceptron(new GradientInfo { Gradient = Gradient.SGD })
            .AddLayer(new DenseInfo(32, Activation.ReLU))
            .AddLayer(new DenseInfo(1, Activation.Sigmoid));

        var optimizer = new Optimizer<MultiLayerPerceptron>(mlp, pair, Loss.MSE);
        await optimizer.Train(epoch => epoch.Index == maxEpoch);
        
        foreach (var (ins, outs) in trainInputs)
        {
            var pred = optimizer.Model.Predict(ins.First());
            Console.WriteLine($"Answer {outs[0][0]} Confidence {pred.Confidence}");
        }
    }
}