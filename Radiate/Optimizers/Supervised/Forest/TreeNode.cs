namespace Radiate.Optimizers.Supervised.Forest;

public class TreeNode
{
    private readonly TreeNode _leftChild;
    private readonly TreeNode _rightChild;
    private readonly float _label;
    
    public TreeNode() { }

    public TreeNode(TreeNode leftChild, TreeNode rightChild, float label = 0)
    {
        _leftChild = leftChild;
        _rightChild = rightChild;
        _label = label;
    }

    public bool IsLeaf => _leftChild == null && _rightChild == null;

}