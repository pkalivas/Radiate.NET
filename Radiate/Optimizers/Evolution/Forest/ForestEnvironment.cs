using Radiate.Activations;
using Radiate.Optimizers.Evolution.Forest.Info;

namespace Radiate.Optimizers.Evolution.Forest;

public class ForestEnvironment : EvolutionEnvironment
{
    public int InputSize { get; set; }
    public float[] OutputCategories { get; set; }
    public int MaxHeight { get; set; } = 5;
    public int StartHeight { get; set; } = 3;
    public int NumTrees { get; set; } = 10;
    public float NodeAddRate { get; set; } = .1f;
    public float ShuffleRate { get; set; } = .05f;
    public OperatorNodeEnvironment OperatorNodeEnvironment { get; set; } = new();
    public NeuronNodeEnvironment NeuronNodeEnvironment { get; set; } = new();
    public SeralTreeNodeType NodeType { get; set; } = SeralTreeNodeType.Operator;
    
    public override void Reset() { }
    
    public override T GenerateGenome<T>()
    {
        var type = typeof(T).Name;
        var info = new SeralForestInfo(NodeType, NeuronNodeEnvironment.Activations, 
            InputSize, OutputCategories, StartHeight, NeuronNodeEnvironment.UseRecurrent, NumTrees);
        
        if (type is nameof(SeralTree))
        {
            return new SeralTree(info) as T;
        }

        if (type is nameof(SeralForest))
        {
            return new SeralForest(info) as T;
        }

        throw new Exception("Cannot make genome");
    }
}

public class OperatorNodeEnvironment
{
    public float SplitValueMutateRate { get; set; } = .1f;
    public float SplitIndexMutateRate { get; set; } = .1f;
    public float OutputCategoryMutateRate { get; set; } = .1f;
    public float OperatorMutateRate { get; set; } = .05f;
}

public class NeuronNodeEnvironment
{
    public bool UseRecurrent { get; set; } = false;
    public float SplitIndexMutateRate { get; set; } = .1f;
    public float OutputCategoryMutateRate { get; set; } = .1f;
    public float WeightMutateRate { get; set; } = .8f;
    public float EditWeights { get; set; } = .1f;
    public float WeightMovementRate { get; set; } = 1.5f;
    public float ActivationMutateRate { get; set; } = .2f;
    public IEnumerable<Activation> Activations { get; set; } = new[] { Activation.Sigmoid };
}