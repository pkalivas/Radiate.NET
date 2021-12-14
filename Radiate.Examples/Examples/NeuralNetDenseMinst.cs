using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Data;
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
            const int featureLimit = 500;
            const double splitPct = .75;
            const int hiddenLayerSize = 256;
            const int maxEpochs = 50;
            const int batchSize = 50;

            var (normalizedInputs, indexedLabels) = await new Mnist(featureLimit).GetDataSet();

            var inputSize = normalizedInputs.Select(input => input.Length).Distinct().Single();
            var outputSize = indexedLabels.Select(target => target.Length).Distinct().Single();

            var splitIndex = (int) (normalizedInputs.Count - (normalizedInputs.Count * splitPct));
            var trainFeatures = normalizedInputs.Skip(splitIndex).ToList();
            var trainTargets = indexedLabels.Skip(splitIndex).ToList();
            var testFeatures = normalizedInputs.Take(splitIndex).ToList();
            var testTargets = indexedLabels.Take(splitIndex).ToList();

            var neuralNetwork = new MultiLayerPerceptron(inputSize, outputSize)
                .AddLayer(new DenseInfo(hiddenLayerSize, Activation.Sigmoid))
                .AddLayer(new DenseInfo(hiddenLayerSize, Activation.SoftMax));

            var optimizer = new Optimizer(neuralNetwork, Loss.CrossEntropy);

            var progressBar = new ProgressBar(maxEpochs);
            await optimizer.Train(trainFeatures, trainTargets, batchSize, (epochs) => 
            {
                var currentEpoch = epochs.Last();
                progressBar.Tick($"Loss: {currentEpoch.Loss} Accuracy: {currentEpoch.ClassificationAccuracy}");
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