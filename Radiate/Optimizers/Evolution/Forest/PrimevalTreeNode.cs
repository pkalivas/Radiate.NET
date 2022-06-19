namespace Radiate.Optimizers.Evolution.Forest;

public class PrimevalTreeNode
{
    public PrimevalTreeNode Parent;
    public PrimevalTreeNode LeftChild;
    public PrimevalTreeNode RightChild;

    private readonly Random _random;
    private readonly Guid _nodeId;
    private readonly int _splitIndex;
    private readonly int _outputCategory;
    private readonly float _splitValue;
    private readonly Operator _operator;

    public PrimevalTreeNode(int inputSize, int outputSize)
    {
        _random = new Random();
        _nodeId = Guid.NewGuid();
        _splitIndex = _random.Next(0, inputSize);
        _outputCategory = _random.Next(0, outputSize);
        _splitValue = _random.NextSingle();
        _operator = (Operator)_random.Next(0, 3);
    }

    public PrimevalTreeNode(PrimevalTreeNode node)
    {
        _random = new Random();
        _nodeId = node.Id;
        _splitIndex = node._splitIndex;
        _outputCategory = node._outputCategory;
        _splitValue = node._splitValue;
        _operator = node._operator;
    }

    public Guid Id => _nodeId;

    public PrimevalTreeNode DeepCopy(PrimevalTreeNode parent)
    {
        var newNode = new PrimevalTreeNode(this);
        if (LeftChild is not null)
        {
            newNode.LeftChild = LeftChild.DeepCopy(newNode);
        }

        if (RightChild is not null)
        {
            newNode.RightChild = RightChild.DeepCopy(newNode);
        }

        newNode.Parent = parent;

        return newNode;
    }

    public int Size()
    {
        var result = 1;
        
        if (IsLeaf())
        {
            return result;
        }
        
        if (LeftChild is not null)
        {
            result += LeftChild.Size();
        }

        if (RightChild is not null)
        {
            result += RightChild.Size();
        }

        return result;
    }

    public void AddRandom(PrimevalTreeNode node)
    {
        var direction = _random.NextDouble() >= .5;
        if (direction)
        {
            if (LeftChild is not null)
            {
                LeftChild.AddRandom(node);
            }
            else
            {
                LeftChild = node;
                node.Parent = LeftChild;
            }
        }
        else
        {
            if (RightChild is not null)
            {
                RightChild.AddRandom(node);
            }
            else
            {
                RightChild = node;
                node.Parent = RightChild;
            }
        }
    }

    public bool IsLeaf() => LeftChild is null && RightChild is null;

    public bool IsLeftChild() => Parent?.LeftChild != null && Parent.LeftChild.Id == _nodeId;
    
    public bool IsRightChild() => Parent?.RightChild != null && Parent.RightChild.Id == _nodeId;
    
    public int Height() => 1 + Math.Max(LeftChild?.Height() ?? 0, RightChild?.Height() ?? 0);

    public int Depth() => Parent is null ? 0 : Parent.Depth() + 1;
}