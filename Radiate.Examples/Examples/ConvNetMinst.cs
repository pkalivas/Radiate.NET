using System.IO;
using Newtonsoft.Json;
using Radiate.Data;
using Radiate.Data.Utils;
using Radiate.Domain.Activation;
using Radiate.Domain.Extensions;
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
        const int batchSize = 32;
        const int maxEpochs = 2;
        const int imagePadding = 1;
        var inputShape = new Shape(28, 28, 1);

        var (rawInputs, rawLabels) = await new Mnist(featureLimit).GetDataSet();
        var normalizedInputs = rawInputs.Normalize();
        var oneHotEncode = rawLabels.OneHotEncode();
        
        var featureTargetPair = new FeatureTargetPair(normalizedInputs, oneHotEncode)
            .Transform(inputShape)
            .Batch(batchSize)
            .Pad(imagePadding)
            .Split();

        var trainData = featureTargetPair.TrainingInputs;
        var testData = featureTargetPair.TestingInputs;

        var neuralNetwork = new MultiLayerPerceptron()
            .AddLayer(new ConvInfo(16, 3))
            .AddLayer(new MaxPoolInfo(16, 3) { Stride = 2 })
            .AddLayer(new FlattenInfo())
            .AddLayer(new DenseInfo(64, Activation.Sigmoid))
            .AddLayer(new DenseInfo(featureTargetPair.OutputSize, Activation.SoftMax));

        var optimizer = new Optimizer(neuralNetwork);
        
        var progressBar = new ProgressBar(maxEpochs);
        await optimizer.Train(trainData, (epoch) => 
        {
            var displayString = $"Loss: {epoch.AverageLoss} Accuracy: {epoch.ClassificationAccuracy}";
            
            progressBar.Tick(displayString);
            return maxEpochs == epoch.Index;
        });
        
        var wrap = optimizer.Save();
        await Save(wrap.MultiLayerPerceptronWrap);
        
        var trainValidation = optimizer.Validate(trainData);
        var testValidation = optimizer.Validate(testData);
        
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