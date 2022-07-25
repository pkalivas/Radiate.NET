using Radiate.Activations;
using Radiate.Optimizers.Evolution.Genomes.Forest.Info;

namespace Radiate.Optimizers.Evolution.Genomes.Forest;

public class ForestEnvironment : EvolutionEnvironment
{
    public int InputSize { get; set; }
    public float[] OutputCategories { get; set; }
    public int MaxHeight { get; set; } = 5;
    public int? StartHeight { get; set; }
    public int NumTrees { get; set; } = 10;
    public float NodeAddRate { get; set; } = .1f;
    public float ShuffleRate { get; set; } = .05f;
    public bool UseRecurrent { get; set; } = false;
    public OperatorNodeSettings OperatorNodeSettings { get; set; } = new();
    public NeuronNodeSettings NeuronNodeSettings { get; set; } = new();
    public SeralTreeNodeType NodeType { get; set; } = SeralTreeNodeType.Operator;
    
    public override void Reset() { }
    
    public override T GenerateGenome<T>()
    {
        var type = typeof(T).Name;

        var startHeight = StartHeight ?? MaxHeight;
        if (startHeight > MaxHeight)
        {
            startHeight = MaxHeight;
        }
        
        var info = new SeralForestInfo(InputSize, OutputCategories, startHeight, UseRecurrent, NumTrees);
        var nodeInfo = NodeType switch
        {
            SeralTreeNodeType.Neuron => (INodeInfo) new NeuronNodeInfo(NeuronNodeSettings.LeafNodeActivation, NeuronNodeSettings.Activations),
            SeralTreeNodeType.Operator => (INodeInfo) new OperatorNodeInfo(),
            _ => throw new Exception($"Node Type is not implemented.")
        };
        
        if (type is nameof(SeralTree))
        {
            return new SeralTree(info, nodeInfo) as T;
        }

        if (type is nameof(SeralForest))
        {
            return new SeralForest(info, nodeInfo) as T;
        }

        throw new Exception("Cannot make genome");
    }
}

public class OperatorNodeSettings
{
    public float SplitValueMutateRate { get; set; } = .1f;
    public float SplitIndexMutateRate { get; set; } = .1f;
    public float OutputCategoryMutateRate { get; set; } = .1f;
    public float OperatorMutateRate { get; set; } = .05f;
}

public class NeuronNodeSettings
{
    public float FeatureIndexMutateRate { get; set; } = .05f;
    public float OutputCategoryMutateRate { get; set; } = .1f;
    public float WeightMutateRate { get; set; } = .8f;
    public float EditWeights { get; set; } = .1f;
    public float WeightMovementRate { get; set; } = .8f;
    public float ActivationMutateRate { get; set; } = .01f;
    public Activation LeafNodeActivation { get; set; } = Activation.ExpSigmoid;
    public IEnumerable<Activation> Activations { get; set; } = new[] { Activation.Sigmoid };
}