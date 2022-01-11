﻿using System.Diagnostics;
using Radiate.Data;
using Radiate.Data.Utils;
using Radiate.Domain.Activation;
using Radiate.Domain.Callbacks;
using Radiate.Domain.Extensions;
using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Domain.Tensors;
using Radiate.Examples.Description;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples;

public class TrainLSTM : IExample
{
    public async Task Run()
    {
        var s = new Stopwatch();
        s.Start();
        
        const int trainEpochs = 500;

        var (inputs, targets) = await new SimpleMemory().GetDataSet();
        
        var pair = new TensorTrainSet(inputs, targets).Batch(targets.Count());
        
        var mlp = new MultiLayerPerceptron(new GradientInfo() { Gradient = Gradient.Adam })
            .AddLayer(new LSTMInfo(16, 16))
            .AddLayer(new DenseInfo(1, Activation.Sigmoid));

        var optimizer = new Optimizer<MultiLayerPerceptron>(mlp, pair, Loss.MSE, new []
        {
            new VerboseTrainingCallback(pair, trainEpochs, false)
        });
        
        var lstm = await optimizer.Train(epoch => epoch.Index == trainEpochs);
        
        s.Stop();
        Console.WriteLine($"\n{s.ElapsedMilliseconds}");

        foreach (var (ins, outs) in pair.TrainingInputs)
        {
            foreach (var (i, j) in ins.Zip(outs))
            {
                var pred = lstm.Predict(i);
                Console.WriteLine($"Input {i.ToArray()[0]} Expecting {j.ToArray()[0]} Guess {pred.Confidence}");    
            }
        }
        
        Console.WriteLine("\nTesting Memory...");
        Console.WriteLine($"Input {1f} Expecting {0f} Guess {lstm.Predict(new float[1] { 1 }.ToTensor()).Confidence}");
        Console.WriteLine($"Input {0f} Expecting {0f} Guess {lstm.Predict(new float[1] { 0 }.ToTensor()).Confidence}");
        Console.WriteLine($"Input {0f} Expecting {0f} Guess {lstm.Predict(new float[1] { 0 }.ToTensor()).Confidence}");
        Console.WriteLine($"Input {0f} Expecting {1f} Guess {lstm.Predict(new float[1] { 0 }.ToTensor()).Confidence}");
    }
}
