using Radiate.Domain.Extensions;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Forest;

public class DecisionTree
{
    private readonly Random _random = new();

    private readonly ForestInfo _info;
    private readonly TreeNode _root;
    
    public DecisionTree(ForestInfo forestInfo, Tensor features, Tensor targets)
    {
        var info = forestInfo with { NFeatures = Math.Min(features.Shape.Width, forestInfo.NFeatures) };
        
        _info = info;
        _root = GrowTree(features, targets);
    }

    public Prediction Predict(Tensor input) => Traverse(input, _root);

    private TreeNode GrowTree(Tensor features, Tensor targets, int depth = 0)
    {
        var (minSplit, maxDepth, nFeatures) = _info;
        var (_, fWidth, _) = features.Shape;
        
        var labels = targets.Unique();

        if (depth >= maxDepth || labels.Count() == 1 || fWidth < minSplit)
        {
            return new TreeNode(targets);
        }

        var featureIndexes = GetFeatureIndexes(nFeatures);
        var (splitIndex, splitThreshold) = FindSplit(features, targets, featureIndexes);
        
        var (leftIndexes, rightIndexes) = SplitIndexes(features.Column(splitIndex), splitThreshold);
        var (leftFeatures, leftTargets) = SplitTensors(features, targets, leftIndexes);
        var (rightFeatures, rightTargets) = SplitTensors(features, targets, rightIndexes);

        var leftNode = GrowTree(leftFeatures, leftTargets, depth + 1);
        var rightNode = GrowTree(rightFeatures, rightTargets, depth + 1);

        return new TreeNode(leftNode, rightNode, splitIndex, splitThreshold);
    }
    
    private int[] GetFeatureIndexes(int featureNum)
    {
        var indexLookup = new HashSet<int>();
        while (indexLookup.Count < _info.NFeatures)
        {
            var index = _random.Next(0, featureNum);

            if (!indexLookup.Contains(index))
            {
                indexLookup.Add(index);
            }
        }

        return indexLookup.ToArray();
    }
    
    private static Prediction Traverse(Tensor input, TreeNode node)
    {
        while (true)
        {
            if (node.IsLeaf)
            {
                return node.GetPrediction();
            }

            node = node.GetChild(input);
        }
    }

    private static (int splitIndex, float splitThreshold) FindSplit(Tensor features, Tensor targets, int[] featureIndexes)
    {
        var bestGain = float.MinValue;
        var splitIdx = -1;
        var splitThreshold = float.MinValue;
        
        foreach (var index in featureIndexes)
        {
            var column = features.Column(index);
            var unique = column.Unique();
            var (uHeight, _, _) = unique.Shape;

            for (var i = 0; i < uHeight; i++)
            {
                var threshold = unique[i]; 
                var gain = InfoGain(targets, column, threshold);

                if (gain > bestGain)
                {
                    bestGain = gain;
                    splitIdx = index;
                    splitThreshold = threshold;
                }
            }
        }

        return (splitIdx, splitThreshold);
    }

    private static float InfoGain(Tensor targets, Tensor column, float threshold)
    {
        var parentEntropy = targets.HistEntropy();

        var (leftIndexes, rightIndexes) = SplitIndexes(column, threshold);
        var leftThresholds = leftIndexes.Select(idx => targets[idx]).ToArray();
        var rightThresholds = rightIndexes.Select(idx => targets[idx]).ToArray();

        var leftCount = leftThresholds.Length;
        var rightCount = rightThresholds.Length;
        
        if (leftCount == 0 || rightCount == 0)
        {
            return 0;
        }

        var numLeft = Convert.ToSingle(leftCount);
        var entropyLeft = leftThresholds.ToTensor().HistEntropy();
        
        var numRight = Convert.ToSingle(rightCount);
        var entropyRight = rightThresholds.ToTensor().HistEntropy();

        var cNum = Convert.ToSingle(targets.Count());
        var childEntropy = (numLeft / cNum) * entropyLeft + (numRight / cNum) * entropyRight;

        return parentEntropy - childEntropy;
    }

    private static (int[] leftIndexes, int[] rightIndexes) SplitIndexes(Tensor column, float threshold)
    {
        var leftThresholds = column
            .Select((val, idx) => (val, idx))
            .Where(pair => pair.val <= threshold)
            .Select(pair => pair.idx)
            .ToArray();

        var rightThresholds = column
            .Select((val, idx) => (val, idx))
            .Where(pair => pair.val > threshold)
            .Select(pair => pair.idx)
            .ToArray();

        return (leftThresholds, rightThresholds);
    }

    private static (Tensor features, Tensor targets) SplitTensors(Tensor features, Tensor targets, int[] indexes)
    {
        var featureRows = indexes.Select(features.Row).ToArray();

        var newFeatures = Tensor.Stack(featureRows, Axis.Zero);
        var newTargets = indexes.Select(idx => targets[idx]).ToTensor();
        
        return (newFeatures, newTargets);
    }
    
}