using Radiate.Extensions;
using Radiate.Records;
using Radiate.Tensors.Enums;

namespace Radiate.Tensors.Transforms;

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

            if (trainTargets.Any())
            {
                newTargets.Add(trainTargets.Skip(i + layer).Take(1).SelectMany(val => val).ToTensor());
            }
        }

        return (new TrainTestSplit(newFeatures, newTargets), options);
    }

    public Tensor Process(Tensor value, TensorTrainOptions options)
    {
        if (options.Layer == 0)
        {
            return value;
        }

        var rows = value.ToRows();
        var result = new List<Tensor>();
        for (var i = 0; i < value.Shape.Height - options.Layer; i++)
        {
            result.Add(rows.Skip(i).Take(options.Layer).SelectMany(val => val).ToTensor());
        }

        return Tensor.Stack(result.ToArray(), Axis.Zero);
    }
}