using Radiate.Data;
using Radiate.Data.Utils;
using Radiate.Domain.Models;
using Radiate.Domain.Tensors;
using Radiate.Optimizers;
using Radiate.Optimizers.Unsupervised.Clustering;

namespace Radiate.Examples.Examples;

public class BlobMeans : IExample
{
    public async Task Run()
    {
        const int maxEpoch = 100;
        var progressBar = new ProgressBar(maxEpoch);

        var (rawInputs, rawLabels) = await new ClusterBlob().GetDataSet();
        var pair = new TensorTrainSet(rawInputs, rawLabels);

        var kMeans = new KMeans(pair.OutputCategories);
        var optimizer = new Optimizer<KMeans>(kMeans, pair);

        await optimizer.Train(epoch =>
        {
            var displayString = $"Loss: {epoch.AverageLoss} Class Acc: {epoch.ClassificationAccuracy} Reg Acc: {epoch.RegressionAccuracy}";
            progressBar.Tick(displayString);
            return epoch.Index == maxEpoch || epoch.AverageLoss == 0 && epoch.RegressionAccuracy > 0;
        });

        var validator = new Validator();
        var acc = validator.Validate(optimizer.Model, pair.TrainingInputs);

        Console.WriteLine($"\nLoss: {acc.AverageLoss} Accuracy {acc.RegressionAccuracy}");
    }
}