using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Data;
using Radiate.Data.Models;
using Radiate.Data.Utils;
using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Services;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples
{
    public class ConvNetMinst : IExample
    {
        public async Task Run()
        {
            var featureLimit = 100;
            var splitPct = .75;
            var learningRate = 0.1f;
            var hiddenLayerSize = 32;
            var maxEpochs = 500;
            var batchSize = 50;

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
            var kernel = new Kernel(3, 1);
            var imageStride = 1;
            var neuralNetwork = new MultiLayerPerceptron(inputSize, outputSize)
                .AddLayer(new ConvInfo(imageShape, kernel, imageStride, Activation.ReLU))
                .AddLayer(new MaxPoolInfo(kernel, 2))
                .AddLayer(new FlattenInfo())
                .AddLayer(new DenseInfo(Activation.SoftMax));

            var optimizer = new Optimizer(neuralNetwork, Loss.CrossEntropy, imageShape, new GradientInfo
            {
                Gradient = Gradient.Adam,
                LearningRate = 0.001f
            });

            Console.WriteLine("\n\n");
            var progressBar = new ProgressBar(maxEpochs);
            await optimizer.Train(trainFeatures, trainTargets, (epochs) => 
            {
                var currentEpoch = epochs.Last();
                var prevEpoch = epochs.Count > 1 ? epochs.ElementAt(epochs.Count - 2) : currentEpoch;
                
                var lossDiff = Math.Round(currentEpoch.Loss - prevEpoch.Loss, 4);
                var displayString = $"Loss: {currentEpoch.Loss} ({lossDiff}) Accuracy: {currentEpoch.ClassificationAccuracy}";
                
                progressBar.Tick(displayString);
                return maxEpochs == epochs.Count;
            });

            var trainValidation = optimizer.Validate(trainFeatures, trainTargets);
            var testValidation = optimizer.Validate(testFeatures, testTargets);
            
            var trainValid = trainValidation.ClassificationAccuracy;
            var testValid = testValidation.ClassificationAccuracy;
            
            Console.WriteLine($"\nTrain accuracy: {trainValid} - Test accuracy: {testValid}");
        }
    }
}