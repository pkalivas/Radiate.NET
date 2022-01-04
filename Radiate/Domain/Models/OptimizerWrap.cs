using Radiate.Optimizers;

namespace Radiate.Domain.Models;

public class OptimizerWrap
{
    public OptimizerType OptimizerType { get; set; }
    public MultiLayerPerceptronWrap MultiLayerPerceptronWrap { get; set; }
}

public class MultiLayerPerceptronWrap
{
    public List<LayerWrap> LayerWraps { get; set; }
}