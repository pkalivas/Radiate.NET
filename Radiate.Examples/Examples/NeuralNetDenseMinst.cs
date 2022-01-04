﻿using Radiate.Data;
using Radiate.Data.Utils;
using Radiate.Domain.Activation;
using Radiate.Domain.Extensions;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples;

public class NeuralNetDenseMinst : IExample
{
    public async Task Run()
    {
        const int featureLimit = 500;
        const int hiddenLayerSize = 256;
        const int maxEpochs = 50;
        const int batchSize = 50;
        var progressBar = new ProgressBar(maxEpochs);

        var (rawInputs, rawLabels) = await new Mnist(featureLimit).GetDataSet();
        var normalizedInputs = rawInputs.Normalize();
        var oneHotEncode = rawLabels.OneHotEncode();
        
        var featureTargetPair = new TensorPair(normalizedInputs, oneHotEncode)
            .Batch(batchSize)
            .Split();

        var trainData = featureTargetPair.TrainingInputs;
        var testData = featureTargetPair.TestingInputs;
        
        var neuralNetwork = new MultiLayerPerceptron()
            .AddLayer(new DenseInfo(hiddenLayerSize, Activation.Sigmoid))
            .AddLayer(new DenseInfo(featureTargetPair.OutputSize, Activation.SoftMax));

        var optimizer = new Optimizer<MultiLayerPerceptron>(neuralNetwork, Loss.CrossEntropy);
        var network = await optimizer.Train(trainData, (epoch) =>
        {
            progressBar.Tick($"Loss: {epoch.AverageLoss} Accuracy: {epoch.ClassificationAccuracy}");
            return maxEpochs == epoch.Index || Math.Abs(epoch.AverageLoss) < .1;
        });
        
        var validator = new Validator(Loss.CrossEntropy);
        var trainValidation = validator.Validate(network, trainData);
        var testValidation = validator.Validate(network, testData);
        
        var trainValid = trainValidation.ClassificationAccuracy;
        var testValid = testValidation.ClassificationAccuracy;
        
        Console.WriteLine($"\nTrain accuracy: {trainValid} - Test accuracy: {testValid}");
    }
}
