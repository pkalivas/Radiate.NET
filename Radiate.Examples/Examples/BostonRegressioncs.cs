using Radiate.Data;
using Radiate.Domain.Activation;
using Radiate.Domain.Callbacks;
using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Loss;
using Radiate.Domain.Tensors;
using Radiate.Domain.Tensors.Enums;
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
        
        var (features, targets) = await new BostonHousing().GetDataSet();
        
        var featureTargetPair = new TensorTrainSet(features, targets)
            .TransformFeatures(Norm.Standardize)
            .Split();
        
        var linearRegressor = new MultiLayerPerceptron()
            .AddLayer(new DenseInfo(outputSize, Activation.Linear));
        
        var optimizer = new Optimizer<MultiLayerPerceptron>(linearRegressor, featureTargetPair, Loss.MSE, new List<ITrainingCallback>()
        {
            new VerboseTrainingCallback(featureTargetPair, maxEpochs, false)
        });

        await optimizer.Train(epoch => maxEpochs == epoch.Index || Math.Abs(epoch.Loss) < .1);
    }

   
}