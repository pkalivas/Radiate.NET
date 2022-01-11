using Radiate.Data;
using Radiate.Domain.Callbacks;
using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Tensors;
using Radiate.Optimizers;
using Radiate.Optimizers.Unsupervised.Clustering;

namespace Radiate.Examples.Examples;

public class BlobMeans : IExample
{
    public async Task Run()
    {
        const int maxEpoch = 1000;

        var (rawInputs, rawLabels) = await new ClusterBlob().GetDataSet();
        var pair = new TensorTrainSet(rawInputs, rawLabels).Split().Batch(1);
        
        var kMeans = new KMeans(pair.OutputCategories);
        var optimizer = new Optimizer<KMeans>(kMeans, pair, new List<ITrainingCallback>
        {
            new VerboseTrainingCallback(pair, maxEpoch),
        });
        
        await optimizer.Train(epoch => epoch.Index == maxEpoch || epoch.Loss == 0);
    }
}