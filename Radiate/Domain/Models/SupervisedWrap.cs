using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Supervised.Forest.Info;

namespace Radiate.Domain.Models;

public class SupervisedWrap
{
    public SupervisedType SupervisedType { get; set; }
    public MultiLayerPerceptronWrap MultiLayerPerceptronWrap { get; set; }
    public RandomForestWrap RandomForestWrap { get; set; }
}

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
