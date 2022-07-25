using Radiate.Records;
using Radiate.Tensors.Enums;

namespace Radiate.Tensors.Transforms;

public static class TensorSetTransforms
{
    private static readonly List<ITensorSetTransform> Transforms = new()
    {
        new KernelTransform(),
        new NormTransform(),
        new ShuffleTransform(),
        new LayerTransform(),
        new ReshapeTransform(),
        new PaddingTransform(),
        new LookForwardTransform(),
        new SplitTransform(),
    };

    public static (TrainTestSplit, TensorTrainOptions) Apply(TrainTestSplit trainTest, TensorTrainOptions options, TrainTest train) =>
        Transforms.Aggregate((trainTest, options), (curr, transform) => transform.Apply(curr.trainTest, curr.options, train));

    public static Tensor Process(Tensor ten, TensorTrainOptions options) =>
        Transforms.Aggregate(ten, (curr, transform) => transform.Process(curr, options));
}