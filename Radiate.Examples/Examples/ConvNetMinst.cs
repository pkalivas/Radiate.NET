using System.IO;
using Newtonsoft.Json;
using Radiate.Data;
using Radiate.Data.Utils;
using Radiate.Domain.Activation;
using Radiate.Domain.Extensions;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples;

public class ConvNetMinst : IExample
{
    public async Task Run()
    {
        const int featureLimit = 5000;
        const int batchSize = 32;
        const int maxEpochs = 25;
        const int imagePadding = 1;
        var inputShape = new Shape(28, 28, 1);

        var (rawInputs, rawLabels) = await new Mnist(featureLimit).GetDataSet();
        var normalizedInputs = rawInputs.Normalize();
        var oneHotEncode = rawLabels.OneHotEncode();
        
        var featureTargetPair = new TensorTrainSet(normalizedInputs, oneHotEncode)
            .Transform(inputShape)
            .Batch(batchSize)
            .Pad(imagePadding)
            .Split();
        
        var neuralNetwork = new MultiLayerPerceptron()
            .AddLayer(new ConvInfo(16, 3))
            .AddLayer(new MaxPoolInfo(16, 3) { Stride = 2 })
            .AddLayer(new FlattenInfo())
            .AddLayer(new DenseInfo(64, Activation.Sigmoid))
            .AddLayer(new DenseInfo(featureTargetPair.OutputSize, Activation.SoftMax));

        var optimizer = new Optimizer<MultiLayerPerceptron>(neuralNetwork, featureTargetPair);
        
        var progressBar = new ProgressBar(maxEpochs);
        await optimizer.Train(epoch => 
        {
            var displayString = $"Loss: {epoch.AverageLoss} Accuracy: {epoch.ClassificationAccuracy}";
            
            progressBar.Tick(displayString);
            return maxEpochs == epoch.Index;
        });
        
        var wrap = optimizer.Model.Save();
        await Save(wrap.MultiLayerPerceptronWrap);
        
        var (trainAcc, testAcc) = optimizer.Validate();
        
        var trainValid = trainAcc.ClassificationAccuracy;
        var testValid = testAcc.ClassificationAccuracy;
        
        Console.WriteLine($"\nTrain accuracy: {trainValid} - Test accuracy: {testValid}");
    }

    private static async Task Save(MultiLayerPerceptronWrap wrap)
    {
        var path = $"C:\\Users\\peter\\Desktop\\Radiate.NET\\Radiate.Examples\\Saves\\convnet.json";
        var content = JsonConvert.SerializeObject(wrap);

        await File.WriteAllTextAsync(path, content);
    }
}