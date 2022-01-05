using Radiate.Domain.Extensions;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Forest;

public class TreeNode
{
    private readonly TreeNode _leftChild;
    private readonly TreeNode _rightChild;
    private readonly Tensor _targets; 
    private readonly int _featureIndex;
    private readonly float _threshold;
    
    public TreeNode(Tensor targets)
    {
        _targets = targets;
    }
    
    public TreeNode(TreeNode leftChild, TreeNode rightChild, int featureIndex, float threshold)
    {
        _leftChild = leftChild;
        _rightChild = rightChild;
        _featureIndex = featureIndex;
        _threshold = threshold;
    }

    public bool IsLeaf => _leftChild == null && _rightChild == null;

    public TreeNode GetChild(Tensor value) =>
        value[_featureIndex] <= _threshold
            ? _leftChild
            : _rightChild;

    public Prediction GetPrediction()
    {
        var classification = _targets
            .GroupBy(val => val)
            .OrderByDescending(val => val.Count())
            .First().Key;

        var confidence = _targets.Sum() / _targets.Count();

        return new Prediction(new[] { confidence }.ToTensor(), (int) classification, confidence);
    }

}