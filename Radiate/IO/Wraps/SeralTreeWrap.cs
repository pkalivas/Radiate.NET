
namespace Radiate.IO.Wraps;

public class SeralForestWrap
{
    public int InputSize { get; init; }
    public int OutputSize { get; init; }
    public int MaxHeight { get; init; }
    public int NumTrees { get; init; }
    public List<ModelWrap> Trees { get; init; }
}

public class SeralTreeWrap
{
    public Guid RootId { get; init; }
    public List<SeralTreeNodeWrap> Nodes { get; init; }
}

public class SeralTreeNodeWrap
{
    public Guid NodeId { get; init; }
    public Guid LeftChildId { get; init; } = Guid.Empty;
    public Guid RightChildId { get; init; } = Guid.Empty;
    public Guid ParentId { get; init; } = Guid.Empty;
    public int SplitIndex { get; init; }
    public float OutputCategory { get; init; }
    public float SplitValue { get; init; }
    public int Operator { get; init; }
}