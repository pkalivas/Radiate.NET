using Radiate.Optimizers.Evolution.Environment;

namespace Radiate.Optimizers.Evolution.Forest;

public class ForestEnvironment : EvolutionEnvironment
{
    public int StartHeight { get; set; } = 2;
    public int MaxHeight { get; set; } = 5;
    public float NodeAddRate { get; set; } = .1f;
    public float GutRate { get; set; } = .05f;
    public float ShuffleRate { get; set; } = .05f;
    public float SplitValueMutateRate { get; set; } = .1f;
    public float SplitIndexMutateRate { get; set; } = .1f;
    public float OutputCategoryMutateRate { get; set; } = .1f;
    
    public override void Reset() { }
}