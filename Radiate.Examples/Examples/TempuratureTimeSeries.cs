using Radiate.Data;
using Radiate.Data.Utils;
using Radiate.Domain.Activation;
using Radiate.Domain.Extensions;
using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples;

public class TempuratureTimeSeries : IExample
{
    public async Task Run()
    {
        const int featureLimit = 100;
        const int maxEpochs = 100;
        const int batchSize = 25;
        const int layerSize = 10;
        var progressBar = new ProgressBar(maxEpochs);

        var (inputs, answers) = await new TempTimeSeries(featureLimit).GetDataSet();
        
        var normalizedFeatures = inputs.Normalize();
        var normalizedTargets = answers.Normalize();
        var pair = new TensorPair(normalizedFeatures, normalizedTargets)
            .Batch(batchSize)
            .Layer(layerSize)
            .Split();

        var trainInputs = pair.TrainingInputs;
        var testInputs = pair.TestingInputs;

        var gradient = new GradientInfo
        {
            Gradient = Gradient.Adam,
            LearningRate = .001f
        };
        var neuralNetwork = new MultiLayerPerceptron(gradient)
            .AddLayer(new LSTMInfo(16, 16))
            .AddLayer(new DenseInfo(1, Activation.Sigmoid));

        var optimizer = new Optimizer<MultiLayerPerceptron>(neuralNetwork);

        var model = await optimizer.Train(trainInputs, epoch =>
        {
            var displayString = $"Loss: {epoch.AverageLoss} Acc: {epoch.RegressionAccuracy}";
            progressBar.Tick(displayString);
            return epoch.Index == maxEpochs;
        });

        var validator = new Validator();
        var trainAcc = validator.Validate(model, trainInputs);
        var testAcc = validator.Validate(model, testInputs);
        
        Console.WriteLine($"Train Accuracy: {trainAcc.RegressionAccuracy} Test Accuracy: {testAcc.RegressionAccuracy}");
        var t = "";
    }
}