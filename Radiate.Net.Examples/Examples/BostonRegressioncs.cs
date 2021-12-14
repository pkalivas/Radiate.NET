using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Net.Data.Utils;
using Radiate.NET.Domain.Activation;
using Radiate.NET.Domain.Gradients;
using Radiate.NET.Domain.Loss;
using Radiate.NET.Domain.Services;
using Radiate.NET.Optimizers;
using Radiate.NET.Optimizers.Perceptrons;
using Radiate.NET.Optimizers.Perceptrons.Info;

namespace Radiate.Net.Examples.Examples
{
    public class BostonRegression : IExample
    {
        public async Task Run()
        {
            var splitPct = .75;
            var learningRate = 0.01f;
            var hiddenLayerSize = 64;
            var outputSize = 1;
            var maxEpochs = 200;

            var (features, labels) = await Utilities.ReadBostonHousing();
            
            var splitIndex = (int) (features.Count - (features.Count * splitPct));

            var normalizedInputs = FeatureService.Standardize(features);
            var rawLabels = labels.Select(lab => lab.ToArray()).ToList();
            var inputSize = normalizedInputs.Select(input => input.Length).Distinct().Single();

            var trainFeatures = normalizedInputs.Skip(splitIndex).ToList();
            var trainTargets = rawLabels.Skip(splitIndex).ToList();
            var testFeatures = normalizedInputs.Take(splitIndex).ToList();
            var testTargets = rawLabels.Take(splitIndex).ToList();

            var linearRegressor = new MultiLayerPerceptron(inputSize, outputSize)
                .AddLayer(new DenseInfo(outputSize, Activation.Linear));
            
            var optimizer = new Optimizer(linearRegressor, Loss.MSE);

            var progressBar = new ProgressBar(maxEpochs);
            await optimizer.Train(trainFeatures, trainTargets, (epochs) => 
            {
                var currentEpoch = epochs.Last();
                var displayString = $"Loss: {currentEpoch.Loss} Accuracy: {currentEpoch.RegressionAccuracy}";
                progressBar.Tick(displayString);
                return maxEpochs == epochs.Count || Math.Abs(currentEpoch.Loss) < .1;
            });
            
            var trainValidation = optimizer.Validate(trainFeatures, trainTargets);
            var testValidation = optimizer.Validate(testFeatures, testTargets);
            
            var trainValid = trainValidation.RegressionAccuracy;
            var testValid = testValidation.RegressionAccuracy;
            
            Console.WriteLine($"Train accuracy: {trainValid} - Test accuracy: {testValid}");
        }

       
    }
}