using Radiate.Domain.Extensions;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors.Enums;

namespace Radiate.Domain.Tensors.Transforms;

public class LayerTransform : ITensorSetTransform
{
    public (TrainTestSplit, TensorTrainOptions) Apply(TrainTestSplit trainTest, TensorTrainOptions options, TrainTest train)
    {
        if (options.Layer == 0)
        {
            return (trainTest, options);
        }
        
        var (trainFeatures, trainTargets) = trainTest;
        var newFeatures = new List<Tensor>();
        var newTargets = new List<Tensor>();
        var layer = options.Layer;
        
        for (var i = 0; i < trainFeatures.Count - layer - 1; i++)
        {
            newFeatures.Add(trainFeatures.Skip(i).Take(layer).SelectMany(val => val).ToTensor());
            newTargets.Add(trainTargets.Skip(i + layer).Take(1).SelectMany(val => val).ToTensor());
        }

        return (new TrainTestSplit(newFeatures, newTargets), options);
    }

    public Tensor Process(Tensor value, TensorTrainOptions options)
    {
        if (options.Layer == 0)
        {
            return value;
        }

        var rows = new List<Tensor>();
        for (var i = 0; i < options.Layer; i++)
        {
            rows.Add(value.Row(i));
        }

        return Tensor.Stack(rows.ToArray(), Axis.One);
    }
}