using Radiate.Extensions;
using Radiate.IO.Wraps;
using Radiate.Optimizers.Evolution.Forest.Info;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Forest;

public class SeralTreeNode
{
    private readonly Random _random = new();
    
    public readonly string Id;
    
    public SeralTreeNode Parent;
    public SeralTreeNode LeftChild;
    public SeralTreeNode RightChild;
    
    private int _splitIndex;
    private float _outputCategory;
    private float _splitValue;
    private Operator _operator;

    private SeralNodeInfo _info = new();

    public SeralTreeNode(int inputSize, float[] outputCategories)
    {
        _splitIndex = _random.Next(0, inputSize);
        _outputCategory = outputCategories[_random.Next(0, outputCategories.Length)];
        _splitValue = _random.NextSingle();
        _operator = (Operator)_random.Next(0, 3);
        
        Id = $"{_splitIndex}_{_outputCategory}_{_operator}";
    }

    public SeralTreeNode(Guid nodeId, IReadOnlyDictionary<Guid, SeralTreeNodeWrap> wraps)
    {
        var node = wraps[nodeId];

        _splitIndex = node.SplitIndex;
        _splitValue = node.SplitValue;
        _outputCategory = node.OutputCategory;
        _operator = (Operator)node.Operator;

        Id = $"{_splitIndex}_{_outputCategory}_{_operator}";

        if (node.LeftChildId != Guid.Empty)
        {
            LeftChild = new SeralTreeNode(node.LeftChildId, wraps)
            {
                Parent = this
            };
        }

        if (node.RightChildId != Guid.Empty)
        {
            RightChild = new SeralTreeNode(node.RightChildId, wraps)
            {
                Parent = this
            };
        }
    }

    public SeralTreeNode(SeralTreeNode node)
    {
        _splitIndex = node._splitIndex;
        _outputCategory = node._outputCategory;
        _splitValue = node._splitValue;
        _operator = node._operator;
        
        Id = $"{_splitIndex}_{_outputCategory}_{_operator}";
    }

    public List<SeralTreeNodeWrap> Save(Guid parentId, Guid nodeId)
    {
        var nodes = new List<SeralTreeNodeWrap>();

        if (IsLeaf)
        {
            nodes.Add(new SeralTreeNodeWrap
            {
                NodeId = nodeId,
                ParentId = parentId,
                SplitIndex = _splitIndex,
                SplitValue = _splitValue,
                OutputCategory = _outputCategory,
                Operator = (int)_operator
            });

            return nodes;
        }
        
        var leftId = Guid.NewGuid();
        var rightId = Guid.NewGuid();

        if (LeftChild is not null)
        {
            nodes.AddRange(LeftChild.Save(nodeId, leftId));
        }

        if (RightChild is not null)
        {
            nodes.AddRange(RightChild.Save(nodeId, rightId));
        }
        
        nodes.Add(new SeralTreeNodeWrap
        {
            NodeId = nodeId,
            LeftChildId = LeftChild is null ? Guid.Empty : leftId,
            RightChildId = RightChild is null ? Guid.Empty : rightId,
            ParentId = parentId,
            Operator = (int)_operator,
            SplitIndex = _splitIndex,
            SplitValue = _splitValue,
            OutputCategory = _outputCategory
        });

        return nodes;
    }

    public Prediction Predict(Tensor input)
    {
        if (IsLeaf)
        {
            return new Prediction(new[] { _splitValue }.ToTensor(), (int)_outputCategory, _outputCategory);
        }

        return _operator switch
        {
            Operator.EqualTo => Propagate(input, ten => ten[_splitIndex] == _splitValue),
            Operator.GreaterThan => Propagate(input, ten => ten[_splitIndex] > _splitValue),
            Operator.LessThan => Propagate(input, ten => ten[_splitIndex] < _splitValue),
            _ => throw new Exception("Operator not implemented")
        };
    }

    private Prediction Propagate(Tensor input, Func<Tensor, bool> func)
    {
        if (func(input) && LeftChild is not null)
        {
            return LeftChild.Predict(input);
        }

        if (RightChild is not null)
        {
            return RightChild.Predict(input);
        }

        if (LeftChild is not null)
        {
            return LeftChild.Predict(input);
        }

        throw new Exception("Failed to find leaf node.");
    }

    public void Gut(int inputSize, float[] outputCategories)
    {
        _splitIndex = _random.Next(0, inputSize);
        _outputCategory = outputCategories[_random.Next(0, outputCategories.Length)];
        _splitValue = _random.NextSingle();
        _operator = (Operator)_random.Next(0, 3);
    }

    public void MutateSplitValue()
    {
        _splitValue += ((float)_random.NextDouble() * 2) - 1;
    }

    public void MutateSplitIndex(int inputSize)
    {
        _splitIndex = _random.Next(0, inputSize);
    }
    
    public void MutateOutputCategory(float[] outputCategories)
    {
        _outputCategory = outputCategories[_random.Next(0, outputCategories.Length)];
    }

    public void MutateOperator()
    {
        _operator = (Operator)_random.Next(0, 3);
    }

    public SeralTreeNode DeepCopy(SeralTreeNode parent)
    {
        var newNode = new SeralTreeNode(this);

        newNode.LeftChild = LeftChild?.DeepCopy(newNode);
        newNode.RightChild = RightChild?.DeepCopy(newNode);

        newNode.Parent = parent;

        return newNode;
    }

    public int Size()
    {
        var result = 1;
        
        if (IsLeaf)
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
    
    public bool IsLeftChild => Parent?.LeftChild != null && Parent.LeftChild.Id == Id;
    
    public bool IsRightChild => Parent?.RightChild != null && Parent.RightChild.Id == Id;

    public void Reset()
    {
        _info = new SeralNodeInfo();
    }

    public int Height()
    {
        var (height, _) = _info;
        if (height.HasValue)
        {
            return height.Value;
        }

        var newHeight = 1 + Math.Max(LeftChild?.Height() ?? 0, RightChild?.Height() ?? 0);
        _info = _info with { Height =  newHeight };
        
        return newHeight;
    }

    public int Depth()
    {
        var (_, depth) = _info;
        if (depth.HasValue)
        {
            return depth.Value;
        }
        
        var newDepth = Parent is null ? 0 : 1 + Parent.Depth();
        _info = _info with { Depth = newDepth };
        
        return newDepth;
    }

    public static SeralTreeNode AddNewNode(Random random, SeralTreeNode parent, SeralTreeNode newNode)
    {
        if (parent == null)
        {
            return newNode;
        }
        
        if (random.NextDouble() < .5)
        {
            parent.LeftChild = AddNewNode(random, parent.LeftChild, newNode);
        }
        else
        {
            parent.RightChild = AddNewNode(random, parent.RightChild, newNode);
        }

        return parent;
    }

    public void Print(int level)
    {
        if (LeftChild is not null)
        {
            LeftChild.Print(level + 1);
        }

        var tabs = string.Join('\t', Enumerable.Range(0, level + 1).Select(_ => ""));
        Console.WriteLine($"{tabs}Node[{Id} :: P={Parent?.Id} :: L={LeftChild?.Id} :: R={RightChild?.Id}]\n");

        if (RightChild is not null)
        {
            RightChild.Print(level + 1);
        }
    }

    private bool IsLeaf => LeftChild == null && RightChild == null;
}