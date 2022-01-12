using Radiate.Optimizers.Supervised.Forest.Info;

namespace Radiate.IO.Wraps;

public class MultiLayerPerceptronWrap
{
    public List<LayerWrap> LayerWraps { get; init; }
}

public class RandomForestWrap
{
    public int NTrees { get; init; }
    public ForestInfo Info { get; init; }
    public List<DecisionTreeWrap> Trees { get; init; }
}

public class SVMWrap
{
    public List<HyperPlaneWrap> HyperPlanes { get; init; }
}
