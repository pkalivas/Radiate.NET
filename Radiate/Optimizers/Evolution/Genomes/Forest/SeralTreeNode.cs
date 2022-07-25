using Radiate.IO.Wraps;
using Radiate.Optimizers.Evolution.Genomes.Forest.Info;
using Radiate.Optimizers.Evolution.Genomes.Forest.Nodes;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Genomes.Forest;

public class SeralTreeNode
{
    private readonly Guid _id = Guid.NewGuid();
    private readonly ISeralTreeNode _node;

    private SeralNodeInfo _info = new();

    public SeralTreeNode Parent;
    public SeralTreeNode LeftChild;
    public SeralTreeNode RightChild;
    
    public SeralTreeNode(ISeralTreeNode node)
    {
        _node = node;
    }

    public SeralTreeNode(Guid nodeId, IReadOnlyDictionary<Guid, SeralTreeNodeWrap> wraps)
    {
        var node = wraps[nodeId];

        if (node.OperatorWrap is not null)
        {
            _node = new OperatorTreeNode(node.OperatorWrap);
        }

        if (node.NeuronTreeNodeWrap is not null)
        {
            _node = new NeuronTreeNode(node.NeuronTreeNodeWrap);
        }
        
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
        _node = node._node switch
        {
            OperatorTreeNode operatorTreeNode => new OperatorTreeNode(operatorTreeNode),
            NeuronTreeNode neuronTreeNode => new NeuronTreeNode(neuronTreeNode)
        };
    }

    public float Weight => _node.Weight();
    
    public int InnovationId => _node.InnovationNumber();
    
    public bool IsLeftChild => Parent?.LeftChild != null && Parent.LeftChild._id == _id;
    
    public bool IsRightChild => Parent?.RightChild != null && Parent.RightChild._id == _id;
    
    public bool IsLeaf => LeftChild == null && RightChild == null;
    
    public List<SeralTreeNodeWrap> Save(Guid parentId, Guid nodeId)
    {
        var nodes = new List<SeralTreeNodeWrap>();

        if (IsLeaf)
        {
            nodes.Add(new SeralTreeNodeWrap
            {
                NodeId = nodeId,
                ParentId = parentId,
                OperatorWrap = _node is OperatorTreeNode operatorNode ? operatorNode.Save() : null,
                NeuronTreeNodeWrap = _node is NeuronTreeNode neuronTreeNode ? neuronTreeNode.Save() : null
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
            OperatorWrap = _node is OperatorTreeNode opNode ? opNode.Save() : null,
            NeuronTreeNodeWrap = _node is NeuronTreeNode neuronNode ? neuronNode.Save() : null
        });

        return nodes;
    }

    public void Mutate(ForestEnvironment environment)
    {
        _node.Mutate(environment);
    }

    public (SeralTreeNode child, Prediction prediction) Propagate(Tensor input, Prediction previousOutput)
    {
        var (direction, prediction) = _node.Propagate(IsLeaf, input, previousOutput);
        if (direction < 0 && LeftChild is not null)
        {
            return (LeftChild, prediction);
        }

        return (RightChild ?? LeftChild, prediction);
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
        Console.WriteLine($"{tabs}Node[{_id} :: P={Parent?._id} :: L={LeftChild?._id} :: R={RightChild?._id}]\n");

        if (RightChild is not null)
        {
            RightChild.Print(level + 1);
        }
    }
}