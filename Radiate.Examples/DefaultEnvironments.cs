using Radiate.Activations;
using Radiate.Optimizers.Evolution.Forest;
using Radiate.Optimizers.Evolution.Neat;

namespace Radiate.Examples;

public static class DefaultEnvironments
{
    public static ForestEnvironment RecurrentNeuronForest => new()
    {
        MaxHeight = 5,
        StartHeight = 5,
        NumTrees = 25,
        NodeAddRate = .05f,
        ShuffleRate = .05f,
        UseRecurrent = true,
        NodeType = SeralTreeNodeType.Neuron,
        NeuronNodeEnvironment = new NeuronNodeEnvironment
        {
            FeatureIndexMutateRate = .1f,
            OutputCategoryMutateRate = .1f,
            WeightMovementRate = 1.5f,
            WeightMutateRate = .8f,
            EditWeights = .1f,
            LeafNodeActivation = Activation.Linear,
            Activations = new[]
            {
                Activation.ExpSigmoid
            }
        }
    };
    
    public static ForestEnvironment NeuronForest => new()
    {
        MaxHeight = 20,
        StartHeight = 5,
        NumTrees = 25,
        NodeAddRate = .05f,
        ShuffleRate = .05f,
        NodeType = SeralTreeNodeType.Neuron,
        UseRecurrent = false,
        NeuronNodeEnvironment = new NeuronNodeEnvironment
        {
            FeatureIndexMutateRate = .05f,
            OutputCategoryMutateRate = .1f,
            WeightMovementRate = 1.1f,
            WeightMutateRate = .8f,
            EditWeights = .1f,
            LeafNodeActivation = Activation.Linear,
            Activations = new[]
            {
                Activation.Linear,
                Activation.Sigmoid,
                Activation.ReLU
            }
        }
    };
    
    public static ForestEnvironment OperatorNodeForest => new()
    {
        MaxHeight = 20,
        StartHeight = 5,
        NumTrees = 25,
        NodeAddRate = .05f,
        ShuffleRate = .05f,
        NodeType = SeralTreeNodeType.Operator,
        OperatorNodeEnvironment = new OperatorNodeEnvironment
        {
            SplitValueMutateRate = .1f,
            SplitIndexMutateRate = .1f,
            OutputCategoryMutateRate = .1f,
            OperatorMutateRate = .05f
        }
    };

    public static NeatEnvironment RecurrentNeatEnvironment => new()
    {
        RecurrentNeuronRate = .95f,
        ReactivateRate = .2f,
        WeightMutateRate = .8f,
        NewEdgeRate = .14f,
        NewNodeRate = .14f,
        EditWeights = .1f,
        WeightPerturb = 1.5f,
        OutputLayerActivation = Activation.ExpSigmoid,
        ActivationFunctions = new List<Activation>
        {
            Activation.ExpSigmoid,
            Activation.ReLU
        }
    };
}