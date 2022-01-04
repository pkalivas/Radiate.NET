using Radiate.Data;
using Radiate.Data.Utils;
using Radiate.Domain.Activation;
using Radiate.Domain.Extensions;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples;

public class BostonRegression : IExample
{
    public async Task Run()
    {
        const int outputSize = 1;
        const int maxEpochs = 200;

        var (features, targets) = await new BostonHousing().GetDataSet();
        
        var normalizedFeatures = features.Standardize();
        var featureTargetPair = new FeatureTargetPair(normalizedFeatures, targets).Split();

        var trainData = featureTargetPair.TrainingInputs;
        var testData = featureTargetPair.TestingInputs;
        
        var linearRegressor = new MultiLayerPerceptron()
            .AddLayer(new DenseInfo(outputSize, Activation.Linear));
        
        var optimizer = new Optimizer(linearRegressor, Loss.MSE);
        
        var progressBar = new ProgressBar(maxEpochs);
        await optimizer.Train(trainData, (epoch) => 
        {
            var displayString = $"Loss: {epoch.AverageLoss} Accuracy: {epoch.RegressionAccuracy}";
            progressBar.Tick(displayString);
            return maxEpochs == epoch.Index || Math.Abs(epoch.AverageLoss) < .1;
        });
        
        var trainValidation = optimizer.Validate(trainData);
        var testValidation = optimizer.Validate(testData);
        
        var trainValid = trainValidation.RegressionAccuracy;
        var testValid = testValidation.RegressionAccuracy;
        
        Console.WriteLine($"Train accuracy: {trainValid} - Test accuracy: {testValid}");
    }

   
}