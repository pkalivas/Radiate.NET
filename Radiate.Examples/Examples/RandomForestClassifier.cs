using Radiate.Data;
using Radiate.Domain.Callbacks;
using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Tensors;
using Radiate.Examples.Callbacks;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.Forest;
using Radiate.Optimizers.Supervised.Forest.Info;

namespace Radiate.Examples.Examples;

public class RandomForestClassifier : IExample
{
    public async Task Run()
    {
        const int numTrees = 3;
        const int maxDepth = 10;
        const int minSampleSplit = 2;
        
        var (rawFeatures, rawLabels) = await new BreastCancer().GetDataSet();
        var pair = new TensorTrainSet(rawFeatures, rawLabels).Split();
        
        var forest = new RandomForest(numTrees, new ForestInfo(minSampleSplit, maxDepth));
        var optimizer = new Optimizer<RandomForest>(forest, pair, new List<ITrainingCallback>
        {
            new VerboseTrainingCallback(pair),
            new ModelWriterCallback()
        });

        await optimizer.Train();
    }
}