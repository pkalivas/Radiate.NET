using System;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Data;
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
    public class BostonRegression : IExample
    {
        public async Task Run()
        {
            var splitPct = .75;
            var outputSize = 1;
            var maxEpochs = 200;

            var (normalizedInputs, labels) = await new BostonHousing().GetDataSet();
            
            var splitIndex = (int) (normalizedInputs.Count - (normalizedInputs.Count * splitPct));
            var rawLabels = labels.Select(lab => lab.ToArray()).ToList();
            var inputSize = normalizedInputs.Select(input => input.Length).Distinct().Single();

            var trainFeatures = normalizedInputs.Skip(splitIndex).ToList();
            var trainTargets = rawLabels.Skip(splitIndex).ToList();
            var testFeatures = normalizedInputs.Take(splitIndex).ToList();
            var testTargets = rawLabels.Take(splitIndex).ToList();

            var linearRegressor = new MultiLayerPerceptron(inputSize, outputSize)
                .AddLayer(new DenseInfo(Activation.Linear));
            
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