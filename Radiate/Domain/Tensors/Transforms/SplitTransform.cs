using Radiate.Domain.Records;
using Radiate.Domain.Tensors.Enums;

namespace Radiate.Domain.Tensors.Transforms;

public class SplitTransform : ITensorSetTransform
{
    public (TrainTestSplit, TensorTrainOptions) Apply(TrainTestSplit trainTest, TensorTrainOptions options, TrainTest train)
    {
        if (options.SplitPct == 0f)
        {
            return (trainTest, options);
        }
        
        var (trainFeatures, trainTargets) = trainTest;
        var splitIndex = (int) (trainFeatures.Count - (trainFeatures.Count * options.SplitPct));

        var features = train is TrainTest.Train 
            ? trainFeatures.Skip(splitIndex).ToList() 
            : trainFeatures.Take(splitIndex).ToList();
        
        var targets = train is TrainTest.Train
            ? trainTargets.Skip(splitIndex).ToList()
            : trainTargets.Take(splitIndex).ToList();
        
        return (new TrainTestSplit(features, targets), options);
    }

    public Tensor Process(Tensor value, TensorTrainOptions options) => value;
}