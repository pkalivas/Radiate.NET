﻿using Radiate.Data;
using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples;

public class TrainDense : IExample
{
    public async Task Run()
    {
        var (inputs, targets) = await new XOR().GetDataSet();

        var gradient = new GradientInfo { Gradient = Gradient.SGD };
        
        var mlp = new MultiLayerPerceptron()
            .AddLayer(new DenseInfo(32, Activation.ReLU))
            .AddLayer(new DenseInfo(1, Activation.Sigmoid));
        
        var classifier = new Optimizer(mlp, Loss.MSE, new Shape(2), gradient);
        
        await classifier.Train(inputs, targets, (epoch) => epoch.Count == 500);
        
        foreach (var (ins, outs) in inputs.Zip(targets))
        {
            var pred = classifier.Predict(ins);
            Console.WriteLine($"Answer {outs[0]} Confidence {pred.Confidence}");
        }
    }
}