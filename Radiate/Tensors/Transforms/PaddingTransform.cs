using Radiate.Records;
using Radiate.Tensors.Enums;

namespace Radiate.Tensors.Transforms;

public class PaddingTransform : ITensorSetTransform
{
    public (TrainTestSplit, TensorTrainOptions) Apply(TrainTestSplit trainTest, TensorTrainOptions options, TrainTest train)
    {
        if (options.Padding == 0)
        {
            return (trainTest, options);
        }
        
        return (trainTest with
        {
            Features = trainTest.Features.Select(row => row.Pad(options.Padding)).ToList()
        }, options);
    }

    public Tensor Process(Tensor value, TensorTrainOptions options)
    {
        if (options.Padding == 0)
        {
            return value;
        }

        return value.Pad(options.Padding);
    }
}