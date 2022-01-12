using Radiate.Extensions;
using Radiate.IO.Wraps;
using Radiate.Optimizers.Supervised.Forest.Info;
using Radiate.Records;
using Radiate.Tensors;
using Radiate.Tensors.Enums;

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

    public RandomForest(ModelWrap wrap)
    {
        var forest = wrap.RandomForestWrap;
        _nTrees = forest.NTrees;
        _info = forest.Info;
        _trees = forest.Trees.Select(tree => new DecisionTree(tree)).ToArray();
    }

    public List<(Prediction prediction, Tensor target)> Step(Tensor[] features, Tensor[] targets)
    {
        var (inputs, answers) = MergeBatch(features, targets);

        Parallel.For(0, _nTrees, i =>
        {
            var (featureInputs, targetInputs) = BootstrapData(inputs, answers);
            _trees[i] = new DecisionTree(_info, featureInputs, targetInputs);
        });

        var predictions = new List<(Prediction, Tensor)>();

        foreach (var (x, y) in features.Zip(targets))
        {
            predictions.AddRange(_trees
                .Select(tree => tree.Predict(x))
                .Select(prediction => (prediction, y)));
        }

        return predictions;
    }
    
    public void Update(List<Cost> errors, int epochCount) { }
    
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

    public ModelWrap Save() => new()
    {
        ModelType = ModelType.RandomForest,
        RandomForestWrap = new()
        {
            NTrees = _nTrees,
            Info = _info,
            Trees = _trees.Select(tree => tree.Save()).ToList()
        }
    };

    private static (Tensor features, Tensor targets) MergeBatch(Tensor[] features, Tensor[] targets)
    {
        var featureResult = Tensor.Stack(features, Axis.Zero);
        var targetResult = Tensor.Stack(targets, Axis.Zero).Flatten();
        
        return (featureResult, targetResult);
    }

    private static (Tensor features, Tensor targets) BootstrapData(Tensor features, Tensor targets)
    {
        var random = RandomGenerator.RandomGenerator.Next;
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