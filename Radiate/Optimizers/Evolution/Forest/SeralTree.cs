using System.Collections;
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
    private Dictionary<int, float> _innovationWeightLookup;

    public SeralTree(SeralForestInfo info)
    {
        var (inputSize, outputs, maxHeight, _) = info;
        var nodes = Enumerable.Range(0, (2 * maxHeight) - 1)
            .Select(_ => new SeralTreeNode(inputSize, outputs))
            .ToArray();

        _rootNode = MakeTree(null, nodes);
        _height = _rootNode.Height();
        _size = nodes.Length;
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
        _innovationWeightLookup = this.GroupBy(val => val.InnovationId)
            .ToDictionary(key => key.Key, val => val.Sum(node => node.Weight));
    }

    public SeralTree(SeralTree tree) : base(tree.InnovationId)
    {
        _rootNode = tree._rootNode.DeepCopy(null);
        _height = tree._height;
        _size = tree._size;
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
                Nodes = _rootNode.Save(Guid.Empty, rootId)   
            }
        };
    }

    private void AddRandom(ForestEnvironment treeEnv)
    {
        _rootNode = SeralTreeNode.AddNewNode(Random, _rootNode, new SeralTreeNode(treeEnv.InputSize, treeEnv.OutputCategories));
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

    private void GutRandomNode(ForestEnvironment treeEnv)
    {
        var index = Random.Next(0, _size);
        this.ToArray()[index].Gut(treeEnv.InputSize, treeEnv.OutputCategories);
    }
    
    private void MutateSplitValue(ForestEnvironment treeEnv)
    {
        foreach (var node in this)
        {
            if (Random.NextDouble() < treeEnv.SplitValueMutateRate)
            {
                node.MutateSplitValue();
            }
        }
    }

    private void MutateSplitIndex(ForestEnvironment treeEnv)
    {
        foreach (var node in this)
        {
            if (Random.NextDouble() < treeEnv.SplitIndexMutateRate)
            {
                node.MutateSplitIndex(treeEnv.InputSize);
            }
        } 
    }

    private void MutateOutputCategory(ForestEnvironment treeEnv)
    {
        foreach (var node in this)
        {
            if (Random.NextDouble() < treeEnv.OutputCategoryMutateRate)
            {
                node.MutateOutputCategory(treeEnv.OutputCategories);
            }
        } 
    }

    private void MutateOperator(ForestEnvironment treeEnv)
    {
        foreach (var node in this)
        {
            if (Random.NextDouble() < treeEnv.OperatorMutateRate)
            {
                node.MutateOperator();
            }
        }
    }

    private SeralTreeNode BiasedLevelNode()
    {
        var index = Random.Next(0, _size);
    
        var levels = this.Select(node => node.Depth()).ToArray();
        var filtered = this.Where(node => node.Depth() == levels[index]).ToArray();

        var nodeIndex = Random.Next(0, filtered.Count());
        return filtered[nodeIndex];
    }

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
                child.AddRandom(treeEnv);
            }

            if (Random.NextDouble() < treeEnv.ShuffleRate)
            {
                child.Shuffle();
            }

            if (Random.NextDouble() < treeEnv.GutRate)
            {
                child.GutRandomNode(treeEnv);
            }

            if (Random.NextDouble() < treeEnv.SplitValueMutateRate)
            {
                child.MutateSplitValue(treeEnv);
            }

            if (Random.NextDouble() < treeEnv.SplitIndexMutateRate)
            {
                child.MutateSplitIndex(treeEnv);
            }

            if (Random.NextDouble() < treeEnv.OutputCategoryMutateRate)
            {
                child.MutateOutputCategory(treeEnv);
            }

            if (Random.NextDouble() < treeEnv.OperatorMutateRate)
            {
                child.MutateOperator(treeEnv);
            }
        }
        
        child.ResetGenome();

        child._innovationWeightLookup = child.GroupBy(val => val.InnovationId)
            .ToDictionary(key => key.Key, val => val.Sum(node => node.Weight));

        return child as T;
    }

    public async Task<double> Distance<T>(T other, PopulationControl populationControl)
    {
        var parentTwo = other as SeralTree;
        return await DistanceCalculator.Distance(_innovationWeightLookup, parentTwo._innovationWeightLookup, populationControl);
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
    }

    public Prediction Predict(Tensor input) => _rootNode.Predict(input);

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