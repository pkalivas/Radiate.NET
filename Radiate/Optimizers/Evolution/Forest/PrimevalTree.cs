using System.Collections;
using Radiate.Optimizers.Evolution.Environment;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Forest;

public class PrimevalTree: IGenome, IEvolved, IOptimizerModel, IEnumerable<PrimevalTreeNode>
{
    private PrimevalTreeNode _rootNode;
    
    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly int _maxHeight;

    public PrimevalTree(int inputSize, int outputSize, int maxHeight)
    {
        var nodes = Enumerable.Range(0, (2 * maxHeight) - 1)
            .Select(_ => new PrimevalTreeNode(inputSize, outputSize))
            .ToArray();

        _rootNode = MakeTree(null, nodes);
        _inputSize = inputSize;
        _outputSize = outputSize;
        _maxHeight = maxHeight;
    }

    public PrimevalTree(PrimevalTree tree)
    {
        _rootNode = tree._rootNode.DeepCopy(null);
        _inputSize = tree._inputSize;
        _outputSize = tree._outputSize;
        _maxHeight = tree._maxHeight;
    }

    public void AddRandom()
    {
        _rootNode.AddRandom(new PrimevalTreeNode(_inputSize, _outputSize));
    }

    public void Replace(PrimevalTreeNode one, PrimevalTreeNode two)
    {
        if (one.Parent is null)
        {
            return;
        }

        if (one.IsLeftChild())
        {
            var parent = one.Parent;
            parent.LeftChild = two;
            two.Parent = parent;
        }

        if (one.IsRightChild())
        {
            var parent = one.Parent;
            parent.RightChild = two;
            two.Parent = parent;
        }
    }

    private void Balance()
    {
        _rootNode = MakeTree(null, this.ToArray());
    }
    

    private PrimevalTreeNode BiasedLevelNode()
    {
        var random = new Random();
        var height = _rootNode.Height();
        var index = random.Next(0, _rootNode.Size());
        
        var levels = this.Select(node => height - node.Height()).ToArray();
        var filtered = this.Where(node => node.Depth() == levels[index]).ToArray();
        
        var nodeIndex = random.Next(0, filtered.Count());
        return filtered[nodeIndex];
    }

    private static PrimevalTreeNode MakeTree(PrimevalTreeNode parent, PrimevalTreeNode[] nodes)
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
    
    public T Crossover<T, TE>(T other, TE environment, double crossoverRate) where T : class, IGenome where TE : EvolutionEnvironment
    {
        var random = new Random();

        var child = CloneGenome<PrimevalTree>();
        var parentTwo = other as PrimevalTree;
        var treeEnv = environment as ForestEnvironment;

        var nodeOne = child.BiasedLevelNode();
        var nodeTwo = parentTwo.BiasedLevelNode();
        while (nodeOne.Depth() + nodeTwo.Height() > treeEnv.MaxHeight)
        {
            nodeOne = child.BiasedLevelNode();
            nodeTwo = parentTwo.BiasedLevelNode();
        }

        if (random.NextDouble() < crossoverRate)
        {
            child.Replace(nodeOne, nodeTwo);
        }
        else
        {
            if (random.NextDouble() < treeEnv.NodeAddRate)
            {
                child.AddRandom();
            }

            if (random.NextDouble() < treeEnv.ShuffleRate)
            {
                child.Shu
            }
        }

        return child as T;
    }

    public Task<double> Distance<T, TE>(T other, TE environment)
    {
        var parentTwo = other as PrimevalTree;
        var parentTwoLookup = parentTwo.Select(val => val.Id).ToHashSet();

        var totalSame = this.Where(node => parentTwoLookup.Contains(node.Id)).Sum(node => 1.0);
        var totalNodesInBoth = _rootNode.Size() + parentTwoLookup.Count;

        return Task.FromResult(totalSame / (double)totalNodesInBoth);
    }

    public T CloneGenome<T>() where T : class => new PrimevalTree(this) as T;

    public void ResetGenome()
    {
        throw new NotImplementedException();
    }

    public T Randomize<T>() where T : class
    {
        throw new NotImplementedException();
    }

    public Prediction Predict(Tensor input)
    {
        throw new NotImplementedException();
    }

    public IEnumerator<PrimevalTreeNode> GetEnumerator()
    {
        var currentNode = _rootNode;
        var stack = new Stack<PrimevalTreeNode>();
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