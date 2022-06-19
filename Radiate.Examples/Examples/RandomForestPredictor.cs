using Radiate.Callbacks;
using Radiate.Callbacks.Interfaces;
using Radiate.Data;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.Forest;
using Radiate.Optimizers.Supervised.Forest.Info;
using Radiate.Tensors;

namespace Radiate.Examples.Examples;

public class RandomForestPredictor : IExample
{
    public async Task Run()
    {
        const int numTrees = 10;
        const int maxDepth = 10;
        const int minSampleSplit = 2;
        
        var (rawFeatures, rawLabels) = await new IrisFlowers().GetDataSet();
        var pair = new TensorTrainSet(rawFeatures, rawLabels)
            .Shuffle()
            .Split()
            .Batch(rawFeatures.Count);
        
        var forest = new RandomForest(numTrees, new ForestInfo(minSampleSplit, maxDepth));
        var optimizer = new Optimizer(forest, pair, new List<ITrainingCallback>
        {
            new VerboseTrainingCallback(pair),
            new ModelWriterCallback(),
            new ConfusionMatrixCallback()
        });

        await optimizer.Train<RandomForest>();
    }
}