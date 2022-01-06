using Radiate.Domain.Extensions;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Supervised.Forest.Info;

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

    public RandomForest(SupervisedWrap wrap)
    {
        var forest = wrap.RandomForestWrap;
        _nTrees = forest.NTrees;
        _info = forest.Info;
        _trees = forest.Trees.Select(tree => new DecisionTree(tree)).ToArray();
    }

    public void Train(List<Batch> data, LossFunction lossFunction, Func<Epoch, bool> trainFunc)
    {
        var (features, targets) = MergeBatch(data);

        Parallel.For(0, _nTrees, i =>
        {
            var (featureInputs, targetInputs) = BootstrapData(features, targets);
            _trees[i] = new DecisionTree(_info, featureInputs, targetInputs);
        });

        var predictions = new List<(Prediction, Tensor)>();
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
                    predictions.Add((prediction, y));
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

    public SupervisedWrap Save() => new()
    {
        SupervisedType = SupervisedType.RandomForest,
        RandomForestWrap = new()
        {
            NTrees = _nTrees,
            Info = _info,
            Trees = _trees.Select(tree => tree.Save()).ToList()
        }
    };

    private static (Tensor features, Tensor targets) MergeBatch(IReadOnlyCollection<Batch> data)
    {
        var features = data.SelectMany(row => row.Features.Select(ten => ten)).ToArray();
        var targets = data.SelectMany(row => row.Targets.Select(ten => ten)).ToArray();

        var featureResult = Tensor.Stack(features, Axis.Zero);
        var targetResult = Tensor.Stack(targets, Axis.Zero).Flatten();
        
        return (featureResult, targetResult);
    }

    private static (Tensor features, Tensor targets) BootstrapData(Tensor features, Tensor targets)
    {
        var random = new Random();
        var (height, _, _) = features.Shape;

        var newFeatures = new Tensor[height];
        var resultTargets = Tensor.Like(targets.Shape);

        for (var i = 0; i < height; i++)
        {
            var index = random.Next(0, height);
            newFeatures[i] = features.Row(index);
            resultTargets[i] = targets[index];
        }

        var resultFeatures = Tensor.Stack(newFeatures, Axis.Zero);
        
        return (resultFeatures, resultTargets);
    }
}