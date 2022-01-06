using Radiate.Data;
using Radiate.Domain.Tensors;
using Radiate.Examples.Writer;
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
        var optimizer = new Optimizer<RandomForest>(forest, pair);

        var model = await optimizer.Train();
        
        await ModelWriter.Write(model);
        
        var (trainValid, testValid) = optimizer.Validate();
        Console.WriteLine($"Train: {trainValid}\nTest: {testValid}");
    }
}