using Radiate.Domain.Records;
using Radiate.Domain.Tensors.Enums;

namespace Radiate.Domain.Tensors.Transforms;

public interface ITensorSetTransform
{
    (TrainTestSplit, TensorTrainOptions) Apply(TrainTestSplit testTrain, TensorTrainOptions options, TrainTest train);
    Tensor Process(Tensor value, TensorTrainOptions options);
}