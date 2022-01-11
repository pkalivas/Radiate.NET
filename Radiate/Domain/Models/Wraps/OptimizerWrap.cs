using Radiate.Domain.Records;
using Radiate.Optimizers;

namespace Radiate.Domain.Models.Wraps;

public class OptimizerWrap
{
    public TensorTrainOptions TensorOptions { get; set; }
    public Loss.Loss LossFunction { get; set; }
    public ModelWrap ModelWrap { get; set; }
}

public class ModelWrap
{
    public ModelType ModelType { get; set; }
    public MultiLayerPerceptronWrap MultiLayerPerceptronWrap { get; set; }
    public RandomForestWrap RandomForestWrap { get; set; }
    public SVMWrap SVMWrap { get; set; }
    public KMeansWrap KMeansWrap { get; set; }
}