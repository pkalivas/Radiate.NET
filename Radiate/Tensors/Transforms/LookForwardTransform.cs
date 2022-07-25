using Radiate.Records;
using Radiate.Tensors.Enums;

namespace Radiate.Tensors.Transforms;

public class LookForwardTransform : ITensorSetTransform
{
    public (TrainTestSplit, TensorTrainOptions) Apply(TrainTestSplit testTrain, TensorTrainOptions options, TrainTest train)
    {
        if (options.LookForward == 0)
        {
            return (testTrain, options);
        }

        var newFeatures = testTrain.Features.Take(testTrain.Features.Count - options.LookForward).ToList();
        var newTargets = testTrain.Targets.Skip(options.LookForward).ToList();

        return (new TrainTestSplit(newFeatures, newTargets), options);
    }

    public Tensor Process(Tensor value, TensorTrainOptions options) => value;
}