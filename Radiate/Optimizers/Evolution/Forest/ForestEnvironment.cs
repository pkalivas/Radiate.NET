using Radiate.Optimizers.Evolution.Forest.Info;

namespace Radiate.Optimizers.Evolution.Forest;

public class ForestEnvironment : EvolutionEnvironment
{
    public int InputSize { get; set; }
    public float[] OutputCategories { get; set; }
    public int MaxHeight { get; set; } = 5;
    public int NumTrees { get; set; } = 10;
    public float NodeAddRate { get; set; } = .1f;
    public float GutRate { get; set; } = .05f;
    public float ShuffleRate { get; set; } = .05f;
    public float SplitValueMutateRate { get; set; } = .1f;
    public float SplitIndexMutateRate { get; set; } = .1f;
    public float OutputCategoryMutateRate { get; set; } = .1f;
    public float OperatorMutateRate { get; set; } = .05f;
    
    public override void Reset() { }
    
    public override T GenerateGenome<T>()
    {
        var type = typeof(T).Name;
        var info = new SeralForestInfo(InputSize, OutputCategories, MaxHeight, NumTrees);
        
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