using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Forest;

public class DecisionTree
{
    private readonly Random _random = new();

    private readonly ForestInfo _info;
    private readonly TreeNode _root;
    
    public DecisionTree(ForestInfo forestInfo, Batch data)
    {
        _info = forestInfo;
        _root = GrowTree(data);
    }

    private TreeNode GrowTree(Batch data, int depth = 0)
    {
        var (minSplit, maxDepth, nFeatures) = _info;
        var (featureShape, targetShape) = data.InnerShapes;
        var nLabels = data.Targets.SelectMany(row => row.Read1D()).Distinct().Count();

        if (depth >= maxDepth || nLabels == 1 || data.Size < minSplit)
        {
            var leafLabel = data.Targets
                .Select(row => row.Max())
                .OrderByDescending(val => val)
                .First();

            return new TreeNode(null, null, leafLabel);
        }

        var featureIndexes = GetFeatureIndexes(featureShape.Height);

        var bestGain = -1f;
        var splitIdx = -1;
        var splitThreshold = -1f;
        foreach (var index in featureIndexes)
        {
            var column = data.Features.Select(row => row[index]).ToTensor();
            var unique = column.Unique();

            for (var i = 0; i < unique.Shape.Height; i++)
            {
                var threshold = unique[i]; 
                var gain = InfoGain(data.Targets, column, threshold);

                if (gain > bestGain)
                {
                    bestGain = gain;
                    splitIdx = index;
                    splitThreshold = threshold;
                }
            }
        }
        
        return new TreeNode();
    }

    private float InfoGain(Tensor[] targets, Tensor column, float threshold)
    {
        var (cHeight, _, _) = column.Shape;
        var parentEntropy = Entropy(targets);
        
        var leftVals = new List<float>();
        var rightVals = new List<float>();

        for (var i = 0; i < cHeight; i++)
        {
            if (column[i] <= threshold)
            {
                leftVals.Add(column[i]);
            }
            else
            {
                rightVals.Add(column[i]);
            }
        }

        if (!leftVals.Any() || !rightVals.Any())
        {
            return 0;
        }

        var numLeft = Convert.ToSingle(leftVals.Count);
        var entropyLeft = Entropy(leftVals.Select(val => new Tensor(new[] { val })).ToArray());
        
        var numRight = Convert.ToSingle(rightVals.Count);
        var entropyRight = Entropy(rightVals.Select(val => new Tensor(new[] { val })).ToArray());

        var cNum = Convert.ToSingle(cHeight);
        var childEntropy = (numLeft / cNum) * entropyLeft + (numRight / cNum) * entropyRight;

        return parentEntropy - childEntropy;
    }

    private static float Entropy(Tensor[] targets)
    {
        var bins = targets
            .Select(ten => ten.Max())
            .GroupBy(val => val)
            .OrderBy(val => val.Key)
            .Select(group => Convert.ToSingle(group.Count()))
            .ToTensor();

        var psVal = (bins / targets.Length).Read1D();

        return -psVal.Where(val => val > 0).Sum(val => val * (float)Math.Log10(val));
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
}