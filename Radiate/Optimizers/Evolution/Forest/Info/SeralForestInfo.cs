using Radiate.Activations;

namespace Radiate.Optimizers.Evolution.Forest.Info;

public record SeralForestInfo(SeralTreeNodeType NodeType, 
    IEnumerable<Activation> Activation, 
    int InputSize, 
    float[] OutputCategories, 
    int StartHeight, 
    bool UseRecurrent = false,
    int NumTrees = 0);