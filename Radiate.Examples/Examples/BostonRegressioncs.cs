using Radiate.Data;
using Radiate.Data.Utils;
using Radiate.Domain.Activation;
using Radiate.Domain.Extensions;
using Radiate.Domain.Loss;
using Radiate.Domain.Tensors;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples;

public class BostonRegression : IExample
{
    public async Task Run()
    {
        const int outputSize = 1;
        const int maxEpochs = 200;
        const int batchSize = 1;
        
        var (features, targets) = await new BostonHousing().GetDataSet();
        
        var normalizedFeatures = features.Standardize();
        var featureTargetPair = new TensorTrainSet(normalizedFeatures, targets)
            .Batch(batchSize)
            .Split();
        
        var linearRegressor = new MultiLayerPerceptron()
            .AddLayer(new DenseInfo(outputSize, Activation.Linear));
        
        var optimizer = new Optimizer<MultiLayerPerceptron>(linearRegressor, featureTargetPair, Loss.MSE);
        
        var progressBar = new ProgressBar(maxEpochs);
        await optimizer.Train(epoch => 
        {
            var displayString = $"Loss: {epoch.Loss} Accuracy: {epoch.RegressionAccuracy}";
            progressBar.Tick(displayString);
            return maxEpochs == epoch.Index || Math.Abs(epoch.Loss) < .1;
        });

        var (trainAcc, testAcc) = optimizer.Validate();

        var trainValid = trainAcc.RegressionAccuracy;
        var testValid = testAcc.RegressionAccuracy;
        
        Console.WriteLine($"Train accuracy: {trainValid} - Test accuracy: {testValid}");
    }

   
}