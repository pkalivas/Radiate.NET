using Radiate.Optimizers.Supervised.Forest.Info;

namespace Radiate.IO.Wraps;

public class DecisionTreeWrap
{
    public ForestInfo Info { get; init; }
    public Guid RootId { get; init; }
    public List<TreeNodeWrap> Nodes { get; init; } 
}

public class TreeNodeWrap
{
    public Guid NodeId { get; init; }
    public Guid LeftChildId { get; init; } = Guid.Empty;
    public Guid RightChildId { get; init; } = Guid.Empty;
    public float Classification { get; init; }
    public float Confidence { get; init; }
    public int FeatureIndex { get; init; }
    public float Threshold { get; init; }
}