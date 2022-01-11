using Radiate.Optimizers.Supervised.Forest.Info;

namespace Radiate.Domain.Models.Wraps;

public class DecisionTreeWrap
{
    public ForestInfo Info { get; set; }
    public Guid RootId { get; set; }
    public List<TreeNodeWrap> Nodes { get; set; } 
}

public class TreeNodeWrap
{
    public Guid NodeId { get; set; }
    public Guid LeftChildId { get; set; } = Guid.Empty;
    public Guid RightChildId { get; set; } = Guid.Empty;
    public float Classification { get; set; }
    public float Confidence { get; set; }
    public int FeatureIndex { get; set; }
    public float Threshold { get; set; }
}