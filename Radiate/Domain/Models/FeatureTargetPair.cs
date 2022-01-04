using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Models;

public class FeatureTargetPair
{
    private IEnumerable<Tensor> TrainFeatures { get; set; }
    private IEnumerable<Tensor> TrainTargets { get; set; }
    private IEnumerable<Tensor> TestFeatures { get; set; } = new List<Tensor>();
    private IEnumerable<Tensor> TestTargets { get; set; } = new List<Tensor>();
    private int BatchSize { get; set; } = 1;
    private int Padding { get; set; } = 0;

    public FeatureTargetPair(IEnumerable<float[]> features, IEnumerable<float[]> targets)
    {
        TrainFeatures = features.Select(row => row.ToTensor());
        TrainTargets = targets.Select(row => row.ToTensor());
    }

    public List<Batch> TrainingInputs => GetBatches(TrainFeatures, TrainTargets, BatchSize, Padding);

    public List<Batch> TestingInputs => GetBatches(TestFeatures, TestTargets, 1, Padding);

    public int OutputSize => TestTargets.First().Shape.Height;

    public FeatureTargetPair Batch(int batchSize)
    {
        BatchSize = batchSize;
        return this;
    }
    
    public FeatureTargetPair Split(float splitPct = .75f)
    {
        var splitIndex = (int) (TrainFeatures.Count() - (TrainFeatures.Count() * splitPct));

        TrainFeatures = TrainFeatures.Skip(splitIndex).ToList();
        TrainTargets = TrainTargets.Skip(splitIndex).ToList();
        TestFeatures = TrainFeatures.Take(splitIndex).ToList();
        TestTargets = TrainTargets.Take(splitIndex).ToList();

        return this;
    }

    public FeatureTargetPair Transform(Shape shape)
    {
        TrainFeatures = TrainFeatures.Select(row => row.Reshape(shape));

        if (TestFeatures.Any())
        {
            TestFeatures = TestFeatures.Select(row => row.Reshape(shape));
        }
        
        return this;
    }

    public FeatureTargetPair Pad(int padding)
    {
        Padding = padding;
        return this;
    }

    private static List<Batch> GetBatches(IEnumerable<Tensor> features, IEnumerable<Tensor> targets, int batchSize, int padding)
    {
        var batches = new List<Batch>();
        for (var i = 0; i < features.Count(); i += batchSize)
        {
            var batchFeatures = features
                .Skip(i)
                .Take(batchSize)
                .Select(row => row.Pad(padding, padding))
                .ToList();
            
            var batchTargets = targets
                .Skip(i)
                .Take(batchSize)
                .ToList();
            
            batches.Add(new(batchFeatures, batchTargets));
        }

        return batches;
    }
}