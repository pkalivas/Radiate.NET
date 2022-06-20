using Radiate.Losses;
using Radiate.Optimizers;
using Radiate.Records;

namespace Radiate.IO.Wraps;

public class OptimizerWrap
{
    public TensorTrainOptions TensorOptions { get; init; }
    public Loss LossFunction { get; init; }
    public ModelWrap ModelWrap { get; init; }
}

public class ModelWrap
{
    public ModelType ModelType { get; init; }
    public MultiLayerPerceptronWrap MultiLayerPerceptronWrap { get; init; }
    public RandomForestWrap RandomForestWrap { get; init; }
    public SVMWrap SVMWrap { get; init; }
    public KMeansWrap KMeansWrap { get; init; }
    public NeatWrap NeatWrap { get; set; }
    public SeralTreeWrap SeralTreeWrap { get; init; }
    public SeralForestWrap SeralForestWrap { get; init; }
}