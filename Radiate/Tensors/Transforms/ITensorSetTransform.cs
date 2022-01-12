using Radiate.Records;
using Radiate.Tensors.Enums;

namespace Radiate.Tensors.Transforms;

public interface ITensorSetTransform
{
    (TrainTestSplit, TensorTrainOptions) Apply(TrainTestSplit testTrain, TensorTrainOptions options, TrainTest train);
    Tensor Process(Tensor value, TensorTrainOptions options);
}