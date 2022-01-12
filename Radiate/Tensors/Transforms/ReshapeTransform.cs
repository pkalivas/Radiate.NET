using Radiate.Records;
using Radiate.Tensors.Enums;

namespace Radiate.Tensors.Transforms;

public class ReshapeTransform : ITensorSetTransform
{
    public (TrainTestSplit, TensorTrainOptions) Apply(TrainTestSplit trainTest, TensorTrainOptions options, TrainTest train)
    {
        if (options.FeatureShape is null)
        {
            return (trainTest, options);
        }
        
        return (trainTest with { Features = Reshape(trainTest.Features, options.FeatureShape) }, options);
    }

    public Tensor Process(Tensor value, TensorTrainOptions options)
    {
        if (options.FeatureShape is null)
        {
            return value;
        }

        return value.Reshape(options.FeatureShape);
    }
    
    private static List<Tensor> Reshape(IEnumerable<Tensor> tensors, Shape shape)
    {
        var result = new Tensor[tensors.Count()];
        Parallel.For(0, result.Length, i => result[i] = tensors.ElementAt(i).Reshape(shape));
        return result.ToList();
    }
}