using Radiate.Optimizers.Supervised;

namespace Radiate.Domain.Models;

public class SupervisedWrap
{
    public SupervisedType SupervisedType { get; set; }
    public MultiLayerPerceptronWrap MultiLayerPerceptronWrap { get; set; }
}

public class MultiLayerPerceptronWrap
{
    public List<LayerWrap> LayerWraps { get; set; }
}