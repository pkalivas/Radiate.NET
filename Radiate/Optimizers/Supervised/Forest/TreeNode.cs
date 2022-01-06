using Radiate.Domain.Extensions;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Forest;

public class TreeNode
{
    private readonly TreeNode _leftChild;
    private readonly TreeNode _rightChild;
    private readonly float _classification;
    private readonly float _confidence;
    private readonly int _featureIndex;
    private readonly float _threshold;
    
    public TreeNode(Tensor targets)
    {
        _confidence = targets.Sum() / targets.Count();
        _classification = targets
            .GroupBy(val => val)
            .OrderByDescending(val => val.Count())
            .First().Key;
    }
    
    public TreeNode(TreeNode leftChild, TreeNode rightChild, int featureIndex, float threshold)
    {
        _leftChild = leftChild;
        _rightChild = rightChild;
        _featureIndex = featureIndex;
        _threshold = threshold;
    }

    public TreeNode(Guid nodeId, IReadOnlyDictionary<Guid, TreeNodeWrap> wraps)
    {
        var node = wraps[nodeId];

        _classification = node.Classification;
        _confidence = node.Confidence;
        _featureIndex = node.FeatureIndex;
        _threshold = node.Threshold;

        if (node.LeftChildId != Guid.Empty)
        {
            _leftChild = new TreeNode(node.LeftChildId, wraps);
        }

        if (node.RightChildId != Guid.Empty)
        {
            _rightChild = new TreeNode(node.RightChildId, wraps);
        }
    }

    public bool IsLeaf => _leftChild == null && _rightChild == null;

    public TreeNode GetChild(Tensor value) =>
        value[_featureIndex] <= _threshold
            ? _leftChild
            : _rightChild;

    public Prediction GetPrediction() =>
        new(new[] { _confidence }.ToTensor(), (int) _classification, _confidence);
    
    public List<TreeNodeWrap> Save(Guid nodeId)
    {
        var nodes = new List<TreeNodeWrap>();
        if (IsLeaf)
        {
            nodes.Add(new TreeNodeWrap()
            {
                NodeId = nodeId,
                Classification = _classification,
                Confidence = _confidence,
                FeatureIndex = _featureIndex,
                Threshold = _threshold
            });

            return nodes;
        }

        var leftId = Guid.NewGuid();
        var rightId = Guid.NewGuid();

        if (_leftChild is not null)
        {
            nodes.AddRange(_leftChild.Save(leftId));
        }

        if (_rightChild is not null)
        {
            nodes.AddRange(_rightChild.Save(rightId));
        }
        
        nodes.Add(new()
        {
            NodeId = nodeId,
            LeftChildId = _leftChild is null ? Guid.Empty : leftId,
            RightChildId = _rightChild is null ? Guid.Empty : rightId,
            Classification = _classification,
            Confidence = _confidence,
            FeatureIndex = _featureIndex,
            Threshold = _threshold
        });

        return nodes;
    }
    

}