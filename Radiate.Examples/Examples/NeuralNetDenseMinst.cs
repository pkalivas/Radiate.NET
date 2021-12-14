using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Data.Models;
using Radiate.Data.Utils;
using Radiate.Domain.Activation;
using Radiate.Domain.Loss;
using Radiate.Domain.Services;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples
{
    public class NeuralNetDenseMinst : IExample
    {
        public async Task Run()
        {
            var featureLimit = 100;
            var splitPct = .75;
            var hiddenLayerSize = 64;
            var maxEpochs = 50;
            var batchSize = 50;

            var testFeaturesLocation = $"{Environment.CurrentDirectory}\\DataSets\\Minst\\test.gz";
            var features = (await Utilities.UnzipGZAndLoad<List<MinstImage>>(testFeaturesLocation))
                .Take(featureLimit)
                .ToList();
            
            var rawInputs = features
                .Select(diget => diget.Image.Select(point => (float)point).ToList())
                .ToList();
            var rawLabels = features
                .Select(diget => diget.Label)
                .ToList();

            var normalizedInputs = FeatureService.Normalize(rawInputs);
            var indexedLabels = FeatureService.OneHotEncode(rawLabels);

            var inputSize = normalizedInputs.Select(input => input.Length).Distinct().Single();
            var outputSize = indexedLabels.Select(target => target.Length).Distinct().Single();

            var splitIndex = (int) (features.Count - (features.Count * splitPct));
            var trainFeatures = normalizedInputs.Skip(splitIndex).ToList();
            var trainTargets = indexedLabels.Skip(splitIndex).ToList();
            var testFeatures = normalizedInputs.Take(splitIndex).ToList();
            var testTargets = indexedLabels.Take(splitIndex).ToList();

            var neuralNetwork = new MultiLayerPerceptron(inputSize, outputSize)
                .AddLayer(new DenseInfo(hiddenLayerSize, Activation.Sigmoid))
                .AddLayer(new DropoutInfo())
                .AddLayer(new DenseInfo(hiddenLayerSize, Activation.SoftMax));

            var optimizer = new Optimizer(neuralNetwork, Loss.CrossEntropy);

            var progressBar = new ProgressBar(maxEpochs);
            await optimizer.Train(trainFeatures, trainTargets, (epochs) => 
            {
                var currentEpoch = epochs.Last();
                var displayString = $"Loss: {currentEpoch.Loss} Classification Accuracy: {currentEpoch.ClassificationAccuracy}";
                progressBar.Tick(displayString);
                return maxEpochs == epochs.Count || Math.Abs(currentEpoch.Loss) < .1;
            });

            var trainValidation = optimizer.Validate(trainFeatures, trainTargets);
            var testValidation = optimizer.Validate(testFeatures, testTargets);
            
            var trainValid = trainValidation.ClassificationAccuracy;
            var testValid = testValidation.ClassificationAccuracy;
            
            Console.WriteLine($"\nTrain accuracy: {trainValid} - Test accuracy: {testValid}");
        }
        
    }
}