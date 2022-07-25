using Radiate.Optimizers.Evolution.Genomes.Forest;
using Radiate.Optimizers.Evolution.Genomes.Forest.Info;
using Radiate.Records;

namespace Radiate.IO.Wraps;

public class SeralForestWrap
{
    public SeralForestInfo Info { get; init; }
    public List<ModelWrap> Trees { get; init; }
}

public class SeralTreeWrap
{
    public Guid RootId { get; init; }
    public SeralForestInfo Info { get; init; }
    public Prediction? PreviousOutput { get; init; }
    public SeralTreeNodeType NodeType { get; init; }
    public NeuronNodeInfo NeuronNodeInfo { get; init; }
    public OperatorNodeInfo OperatorNodeInfo { get; init; }
    public List<SeralTreeNodeWrap> Nodes { get; init; }
}

public class SeralTreeNodeWrap
{
    public Guid NodeId { get; init; }
    public Guid LeftChildId { get; init; } = Guid.Empty;
    public Guid RightChildId { get; init; } = Guid.Empty;
    public Guid ParentId { get; init; } = Guid.Empty;
    public OperatorTreeNodeWrap OperatorWrap { get; init; }
    public NeuronTreeNodeWrap NeuronTreeNodeWrap { get; init; }
}

public class OperatorTreeNodeWrap
{
    public int SplitIndex { get; init; }
    public float OutputCategory { get; init; }
    public float SplitValue { get; init; }
    public int Operator { get; init; }
}

public class NeuronTreeNodeWrap
{
    public int[] FeatureIndexes { get; init; }
    public float OutputCategory { get; init; }
    public float[] Weights { get; init; }
    public float Bias { get; init; }
    public int Activation { get; init; }
    public int LeafActivation { get; init; }
}