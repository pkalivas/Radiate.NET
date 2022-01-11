﻿using Radiate.Data;
using Radiate.Domain.Callbacks;
using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Tensors;
using Radiate.Examples.Callbacks;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.Forest;
using Radiate.Optimizers.Supervised.Forest.Info;

namespace Radiate.Examples.Examples;

public class RandomForestPredictor : IExample
{
    public async Task Run()
    {
        const int numTrees = 1;
        const int maxDepth = 10;
        const int minSampleSplit = 2;
        
        var (rawFeatures, rawLabels) = await new IrisFlowers().GetDataSet();
        var pair = new TensorTrainSet(rawFeatures, rawLabels)
            .Shuffle()
            .Split()
            .Batch(rawFeatures.Count);
        
        var forest = new RandomForest(numTrees, new ForestInfo(minSampleSplit, maxDepth));
        var optimizer = new Optimizer<RandomForest>(forest, pair, new List<ITrainingCallback>
        {
            new VerboseTrainingCallback(pair),
            new ModelWriterCallback(),
            new ConfusionMatrixCallback()
        });

        await optimizer.Train();
    }
}