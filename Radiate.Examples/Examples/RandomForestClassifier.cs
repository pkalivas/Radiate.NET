using Radiate.Data;
using Radiate.Domain.Tensors;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.Forest;

namespace Radiate.Examples.Examples;

public class RandomForestClassifier : IExample
{
    public async Task Run()
    {
        const int numTrees = 3;
        const int maxDepth = 10;
        const int minSampleSplit = 2;
        const int nFeatures = 30;
        
        var (rawFeatures, rawLabels) = await new BreastCancer().GetDataSet();
        var pair = new TensorTrainSet(rawFeatures, rawLabels).Split();
        
        var forest = new RandomForest(numTrees, new ForestInfo(minSampleSplit, maxDepth, nFeatures));
        var optimizer = new Optimizer<RandomForest>(forest, pair);

        await optimizer.Train();
        
        var (trainValid, testValid) = optimizer.Validate();
        Console.WriteLine($"Train: {trainValid} Test: {testValid}");
    }
}