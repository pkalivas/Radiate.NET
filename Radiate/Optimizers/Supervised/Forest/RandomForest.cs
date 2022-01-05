using Radiate.Domain.Extensions;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Forest;

public class RandomForest : ISupervised
{
    private readonly int _nTrees;
    private readonly ForestInfo _info;
    private readonly DecisionTree[] _trees;
    
    public RandomForest(int nTrees, ForestInfo info)
    {
        _nTrees = nTrees;
        _info = info;
        _trees = new DecisionTree[nTrees];
    }

    public void Train(List<Batch> data, LossFunction lossFunction, Func<Epoch, bool> trainFunc)
    {
        var (features, targets) = MergeBatch(data);

        Parallel.For(0, _nTrees, i =>
        {
            _trees[i] = new DecisionTree(_info, features, targets);
        });

        var predictions = new List<(Tensor, Tensor)>();
        var epochErrors = new List<float>();
        foreach (var (batchFeature, batchTarget) in data)
        {
            foreach (var (x, y) in batchFeature.Zip(batchTarget))
            {
                foreach (var tree in _trees)
                {
                    var prediction = tree.Predict(x);
                    var cost = lossFunction(prediction.Result, y);
                    
                    epochErrors.Add(cost.Loss);
                    predictions.Add((prediction.Result, y));
                }
            }
        }

        var epoch = Validator.ValidateEpoch(epochErrors, predictions);
        
        trainFunc(epoch);
    }
    
    public Prediction Predict(Tensor input)
    {
        var predictions = _trees.Select(tree => tree.Predict(input)).ToList();
        var confidence = predictions.Select(pred => pred.Confidence).Average();
        var classification = predictions
            .Select(pred => pred.Classification)
            .GroupBy(val => val)
            .OrderByDescending(val => val.Count())
            .First().Key;

        return new Prediction(new[] { confidence }.ToTensor(), classification, confidence);
    }

    private static (Tensor features, Tensor targets) MergeBatch(IReadOnlyCollection<Batch> data)
    {
        var features = data.SelectMany(row => row.Features.Select(ten => ten)).ToArray();
        var targets = data.SelectMany(row => row.Targets.Select(ten => ten)).ToArray();

        var featureResult = Tensor.Stack(features, Axis.Zero);
        var targetResult = Tensor.Stack(targets, Axis.Zero).Flatten();
        
        return (featureResult, targetResult);
    }
}