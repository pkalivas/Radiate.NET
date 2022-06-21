using Radiate.Records;
using Radiate.Tensors.Enums;

namespace Radiate.Tensors.Transforms;

public class ShuffleTransform : ITensorSetTransform
{
    public (TrainTestSplit, TensorTrainOptions) Apply(TrainTestSplit trainTest, TensorTrainOptions options, TrainTest train)
    {
        if (!options.Shuffle)
        {
            return (trainTest, options);
        }

        var rand = RandomGenerator.RandomGenerator.Next;
        var (trainFeatures, trainTargets) = trainTest;
        var newTrainFeatures = new List<Tensor>();
        var newTrainTargets = new List<Tensor>();
            
        var indexLookup = new HashSet<int>();
        while (indexLookup.Count < trainFeatures.Count)
        {
            var idx = rand.Next(0, trainFeatures.Count);

            if (!indexLookup.Contains(idx))
            {
                indexLookup.Add(idx);
                newTrainFeatures.Add(trainFeatures[idx]);

                if (trainTargets.Any())
                {
                    newTrainTargets.Add(trainTargets[idx]);
                }
            }
        }
        
        return (trainTest with { Features = newTrainFeatures, Targets = newTrainTargets }, options);
    }

    public Tensor Process(Tensor value, TensorTrainOptions options) => value;
}