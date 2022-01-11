using Radiate.Optimizers.Supervised.Forest.Info;

namespace Radiate.Domain.Models.Wraps;

public class MultiLayerPerceptronWrap
{
    public List<LayerWrap> LayerWraps { get; set; }
}

public class RandomForestWrap
{
    public int NTrees { get; set; }
    public ForestInfo Info { get; set; }
    public List<DecisionTreeWrap> Trees { get; set; }
}

public class SVMWrap
{
    public List<HyperPlaneWrap> HyperPlanes { get; set; }
}
