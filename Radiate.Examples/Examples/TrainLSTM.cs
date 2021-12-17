using Radiate.Data;
using Radiate.Data.Utils;
using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples;

public class TrainLSTM : IExample
{
    public async Task Run()
    {
        const int trainEpochs = 500;
        
        var (inputs, targets) = await new SimpleMemory().GetDataSet();
        
        var mlp = new MultiLayerPerceptron()
            .AddLayer(new LSTMInfo(16, 16))
            .AddLayer(new DenseInfo(1, Activation.Sigmoid));
        
        var classifier = new Optimizer(mlp, Loss.MSE, new Shape(1), new GradientInfo
        {
            Gradient = Gradient.Adam
        });
        
        var batchSize = targets.Count;
        var progressBar = new ProgressBar(trainEpochs);
        await classifier.Train(inputs, targets, batchSize, (epoch) =>
        {
            var current = epoch.Last();
            var displayString = $"Loss: {current.Loss} Accuracy: {current.RegressionAccuracy}";
            progressBar.Tick(displayString);
            return epoch.Count == trainEpochs;
        });
        
        Console.WriteLine();
        foreach (var (ins, outs) in inputs.Zip(targets))
        {
            var pred = classifier.Predict(ins);
            Console.WriteLine($"Input {ins[0]} Expecting {outs[0]} Guess {pred.Confidence}");
        }
        
        Console.WriteLine("\nTesting Memory...");
        Console.WriteLine($"Input {1f} Expecting {0f} Guess {classifier.Predict(new float[1] { 1 }).Confidence}");
        Console.WriteLine($"Input {0f} Expecting {0f} Guess {classifier.Predict(new float[1] { 0 }).Confidence}");
        Console.WriteLine($"Input {0f} Expecting {0f} Guess {classifier.Predict(new float[1] { 0 }).Confidence}");
        Console.WriteLine($"Input {0f} Expecting {1f} Guess {classifier.Predict(new float[1] { 0 }).Confidence}");
    }
}
