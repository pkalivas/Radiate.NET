using System.IO;
using Newtonsoft.Json;
using Radiate.Data;
using Radiate.Data.Utils;
using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples;

public class ConvNetMinst : IExample
{
    public async Task Run()
    {
        const int featureLimit = 5000;
        const double splitPct = .75;
        const int maxEpochs = 30;

        var (normalizedInputs, indexedLabels) = await new Mnist(featureLimit).GetDataSet();

        var inputSize = normalizedInputs.Select(input => input.Length).Distinct().Single();
        var outputSize = indexedLabels.Select(target => target.Length).Distinct().Single();
        
        Console.WriteLine($"Input size: {inputSize}\nOutput size: {outputSize}");
        Console.WriteLine("\nTotal Loaded Data:");
        MinstDiscriptor.Describe(normalizedInputs, indexedLabels);

        var splitIndex = (int) (normalizedInputs.Count - (normalizedInputs.Count * splitPct));
        var trainFeatures = normalizedInputs.Skip(splitIndex).ToList();
        var trainTargets = indexedLabels.Skip(splitIndex).ToList();
        var testFeatures = normalizedInputs.Take(splitIndex).ToList();
        var testTargets = indexedLabels.Take(splitIndex).ToList();

        Console.WriteLine("Training Data:");
        MinstDiscriptor.Describe(trainFeatures, trainTargets);
        Console.WriteLine("\nTesting Data:");
        MinstDiscriptor.Describe(testFeatures, testTargets);
        
        var imageShape = new Shape(28, 28, 1);
        var neuralNetwork = new MultiLayerPerceptron()
            .AddLayer(new ConvInfo(16, 3))
            .AddLayer(new MaxPoolInfo(16, 3) { Stride = 2})
            .AddLayer(new FlattenInfo())
            .AddLayer(new DenseInfo(64, Activation.Sigmoid))
            .AddLayer(new DenseInfo(outputSize, Activation.SoftMax));

        var optimizer = new Optimizer(neuralNetwork, Loss.Difference, imageShape, new GradientInfo
        {
            Gradient = Gradient.Adam,
            LearningRate = 0.01f
        });

        Console.WriteLine("\n\n");
        var progressBar = new ProgressBar(maxEpochs);
        await optimizer.Train(trainFeatures, trainTargets, 32, (epochs) => 
        {
            var currentEpoch = epochs.Last();
            var prevEpoch = epochs.Count > 1 ? epochs.ElementAt(epochs.Count - 2) : currentEpoch;
            
            var lossDiff = Math.Round(currentEpoch.Loss - prevEpoch.Loss, 4);
            var displayString = $"Loss: {currentEpoch.Loss} ({lossDiff}) Accuracy: {currentEpoch.ClassificationAccuracy}";
            
            progressBar.Tick(displayString);
            return maxEpochs == epochs.Count;
        });

        var wrap = optimizer.Save();
        await Save(wrap.MultiLayerPerceptronWrap);
        
        var trainValidation = optimizer.Validate(trainFeatures, trainTargets);
        var testValidation = optimizer.Validate(testFeatures, testTargets);
        
        var trainValid = trainValidation.ClassificationAccuracy;
        var testValid = testValidation.ClassificationAccuracy;
        
        Console.WriteLine($"\nTrain accuracy: {trainValid} - Test accuracy: {testValid}");
    }

    private static async Task Save(MultiLayerPerceptronWrap wrap)
    {
        var path = $"C:\\Users\\peter\\Desktop\\Radiate.NET\\Radiate.Examples\\Saves\\convnet.json";
        var content = JsonConvert.SerializeObject(wrap);

        await File.WriteAllTextAsync(path, content);
    }
}