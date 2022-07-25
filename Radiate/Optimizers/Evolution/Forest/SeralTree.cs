using System.Collections;
using Radiate.Extensions;
using Radiate.IO.Wraps;
using Radiate.Optimizers.Evolution.Forest.Info;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Forest;

public class SeralTree : Allele, IGenome, IPredictionModel, IEnumerable<SeralTreeNode>
{
    private SeralTreeNode _rootNode;
    private int _height;
    private int _size;
    private Prediction _previousOutput;
    private SeralForestInfo _info;
    private INodeInfo _nodeInfo;
    private Dictionary<int, float> _innovationWeightLookup;

    public SeralTree(SeralForestInfo info, INodeInfo nodeInfo)
    {
        _info = info;
        _nodeInfo = nodeInfo;
        _rootNode = MakeTree(null, Enumerable.Range(0, (2 * info.StartHeight) - 1)
            .Select(index => new SeralTreeNode(CreateNode(index)))
            .ToArray());
        _height = _rootNode.Height();
        _size = _rootNode.Size();
        _previousOutput = new Prediction(Array.Empty<float>().ToTensor(), 0);
        _innovationWeightLookup = this.GroupBy(val => val.InnovationId)
            .ToDictionary(key => key.Key, val => val.Sum(node => node.Weight));
    }

    public SeralTree(ModelWrap wrap)
    {
        var treeWrap = wrap.SeralTreeWrap;
        var nodeLookup = treeWrap.Nodes.ToDictionary(key => key.NodeId);

        _rootNode = new SeralTreeNode(treeWrap.RootId, nodeLookup);
        _height = _rootNode.Height();
        _size = nodeLookup.Count;
        _info = treeWrap.Info;
        _nodeInfo = treeWrap.NodeType switch
        {
            SeralTreeNodeType.Neuron => treeWrap.NeuronNodeInfo,
            SeralTreeNodeType.Operator => treeWrap.OperatorNodeInfo
        };
        _innovationWeightLookup = this.GroupBy(val => val.InnovationId)
            .ToDictionary(key => key.Key, val => val.Sum(node => node.Weight));
    }

    public SeralTree(SeralTree tree) : base(tree.InnovationId)
    {
        _rootNode = tree._rootNode.DeepCopy(null);
        _height = tree._height;
        _size = tree._size;
        _info = tree._info;
        _nodeInfo = tree._nodeInfo;
        _innovationWeightLookup = tree._innovationWeightLookup.ToDictionary(key => key.Key, val => val.Value);
    }

    public Dictionary<int, float> WeightLookup => _innovationWeightLookup;

    public ModelWrap Save()
    {
        var rootId = Guid.NewGuid();
        return new()
        {
            ModelType = ModelType.SeralTree,
            SeralTreeWrap = new SeralTreeWrap
            {
                RootId = rootId,
                Info = _info,
                Nodes = _rootNode.Save(Guid.Empty, rootId),
                NodeType = _nodeInfo switch
                {
                    NeuronNodeInfo => SeralTreeNodeType.Neuron,
                    OperatorNodeInfo => SeralTreeNodeType.Operator
                },
                NeuronNodeInfo = _nodeInfo as NeuronNodeInfo ?? null,
                OperatorNodeInfo = _nodeInfo as OperatorNodeInfo ?? null
            }
        };
    }

    private void AddRandom()
    {
        _rootNode = SeralTreeNode.AddNewNode(Random, _rootNode, new SeralTreeNode(CreateNode(_size)));
    }

    private void Balance()
    {
        _rootNode = MakeTree(null, this.ToArray());
    }

    private void Shuffle()
    {
        var nodes = this.ToList().OrderBy(_ => Random.Next()).ToArray();
        _rootNode = MakeTree(null, nodes);
    }

    private SeralTreeNode BiasedLevelNode()
    {
        var index = Random.Next(0, _size);
    
        var levels = this.Select(node => node.Depth()).ToArray();
        var filtered = this.Where(node => node.Depth() == levels[index]).ToArray();

        var nodeIndex = Random.Next(0, filtered.Count());
        return filtered[nodeIndex];
    }
    
    private ISeralTreeNode CreateNode(int index) => _nodeInfo switch
    {
        NeuronNodeInfo neuronInfo => new NeuronTreeNode(index, _info.InputSize, _info.OutputCategories, neuronInfo),
        OperatorNodeInfo operatorInfo => new OperatorTreeNode(index, _info.InputSize, _info.OutputCategories, operatorInfo),
    };

    private static SeralTreeNode MakeTree(SeralTreeNode parent, SeralTreeNode[] nodes)
    {
        if (nodes.Length == 1)
        {
            var node = nodes.Single();
            node.Parent = parent;
            node.LeftChild = null;
            node.RightChild = null;
            return node;
        }
        
        var midPoint = nodes.Length / 2;
        var left = nodes.Take(midPoint).ToArray();
        var right = nodes.Skip(midPoint + 1).ToArray();

        var currentNode = nodes[midPoint];

        currentNode.LeftChild = left.Any() ? MakeTree(currentNode, left) : null;
        currentNode.RightChild = right.Any() ? MakeTree(currentNode, right) : null;
        
        currentNode.Parent = parent;
        
        return currentNode;
    }
    
    private static void Replace(SeralTreeNode one, SeralTreeNode two)
    {
        if (one.Parent is null)
        {
            return;
        }

        if (one.IsLeftChild)
        {
            var parent = one.Parent;
            parent.LeftChild = two.DeepCopy(parent);
        }

        if (one.IsRightChild)
        {
            var parent = one.Parent;
            parent.RightChild = two.DeepCopy(parent);
        }
    }

    public T Crossover<T, TE>(T other, TE environment, double crossoverRate) where T : class, IGenome where TE : EvolutionEnvironment
    {
        var child = CloneGenome<SeralTree>();
        var parentTwo = other as SeralTree;
        var treeEnv = environment as ForestEnvironment;

        var nodeOne = child.BiasedLevelNode();
        var nodeTwo = parentTwo.BiasedLevelNode();
        while (nodeOne.Depth() + nodeTwo.Height() > treeEnv.MaxHeight)
        {
            nodeOne = child.BiasedLevelNode();
            nodeTwo = parentTwo.BiasedLevelNode();
        }

        if (Random.NextDouble() < crossoverRate)
        {
            Replace(nodeOne, nodeTwo);
        }
        else
        {
            if (Random.NextDouble() < treeEnv.NodeAddRate)
            {
                child.AddRandom();
            }

            if (Random.NextDouble() < treeEnv.ShuffleRate)
            {
                child.Shuffle();
            }

            foreach (var node in this)
            {
                node.Mutate(treeEnv);
            }   
        }

        child.ResetGenome();

        child._innovationWeightLookup = child.GroupBy(val => val.InnovationId)
            .ToDictionary(key => key.Key, val => val.Sum(node => node.Weight));

        return child as T;
    }

    public double Distance<T>(T other, DistanceTunings distanceControl)
    {
        var parentTwo = other as SeralTree;
        return DistanceCalculator.Distance(_innovationWeightLookup, parentTwo._innovationWeightLookup, distanceControl);
    }

    public T CloneGenome<T>() where T : class => new SeralTree(this) as T;

    public void ResetGenome()
    {
        var nodes = this.ToArray();
        foreach (var node in nodes)
        {
            node.Reset();
        }

        _height = _rootNode.Height();
        _size = nodes.Length;
        _previousOutput = null;
    }

    public Prediction Predict(Tensor input)
    {
        var currentNode = _rootNode;
        var currentOutput = _previousOutput;

        while (true)
        {
            var outputInput = _info.UseRecurrent ? currentOutput : null;
            var (childNode, childOutput) = currentNode.Propagate(input, outputInput);

            if (currentNode.IsLeaf)
            {
                _previousOutput = childOutput;
                break;
            }

            currentNode = childNode;
            currentOutput = childOutput;
        }

        return _previousOutput;
    }

    public IEnumerator<SeralTreeNode> GetEnumerator()
    {
        var currentNode = _rootNode;
        var stack = new Stack<SeralTreeNode>();
        while (stack.Any() || currentNode != null)
        {
            if (currentNode != null)
            {
                stack.Push(currentNode);
                currentNode = currentNode.LeftChild;
            }
            else
            {
                currentNode = stack.Pop();
                yield return currentNode;
                currentNode = currentNode.RightChild;
            }
        }
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}