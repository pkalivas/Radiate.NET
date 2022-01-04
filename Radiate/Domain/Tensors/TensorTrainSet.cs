using Radiate.Domain.Records;

namespace Radiate.Domain.Tensors;

public class TensorTrainSet
{
    private TrainTestSplit TrainTest { get; set; }
    private TensorTrainOptions Options { get; set; }

    public TensorTrainSet(IEnumerable<float[]> features, IEnumerable<float[]> targets)
    {
        var inputs = features.Select(row => row.ToTensor()).ToList();
        var answers = targets.Select(row => row.ToTensor()).ToList();

        Options = new TensorTrainOptions();
        TrainTest = new TrainTestSplit(inputs, answers, new List<Tensor>(), new List<Tensor>());
    }

    public List<Batch> TrainingInputs => TrainingBatches();

    public List<Batch> TestingInputs => TestingBatches();
    
    public int OutputSize => TrainTest.TrainTargets.First().Shape.Height;

    public int OutputCategories => TrainTest.TrainTargets.Concat(TrainTest.TestTargets)
        .SelectMany(val => val.Read1D())
        .Distinct()
        .Count();

    public TensorTrainSet Batch(int batchSize)
    {
        Options = Options with { BatchSize = batchSize };
        return this;
    }

    public TensorTrainSet Layer(int layer)
    {
        Options = Options with { Layer = layer };
        return this;
    }
    
    public TensorTrainSet Split(float splitPct = .75f)
    {
        Options = Options with { SplitPct = splitPct };
        return this;
    }

    public TensorTrainSet Transform(Shape shape)
    {
        Options = Options with { FeatureShape = shape };
        return this;
    }

    public TensorTrainSet Pad(int padding)
    {
        Options = Options with { Padding = padding };
        return this;
    }

    private List<Batch> TrainingBatches()
    {
        var data = ApplyOptions();
        
        var (size, _, _, _, _) = Options;
        var batchSize = size == 0 ? data.TrainFeatures.Count : size;
        
        var batches = new List<Batch>();
        for (var i = 0; i < data.TrainFeatures.Count; i += batchSize)
        {
            var batchFeatures = data.TrainFeatures
                .Skip(i)
                .Take(batchSize)
                .ToList();
            
            var batchTargets = data.TrainTargets
                .Skip(i)
                .Take(batchSize)
                .ToList();
            
            batches.Add(new Batch(batchFeatures, batchTargets));
        }

        return batches;
    }

    private List<Batch> TestingBatches()
    {
        var data = ApplyOptions();
        
        var batches = new List<Batch>();
        for (var i = 0; i < data.TestFeatures.Count; i++)
        {
            var batchFeatures = data.TestFeatures
                .Skip(i)
                .Take(1)
                .ToList();
            
            var batchTargets = data.TestTargets
                .Skip(i)
                .Take(1)
                .ToList();
            
            batches.Add(new Batch(batchFeatures, batchTargets));
        }

        return batches;
    }

    private TrainTestSplit ApplyOptions()
    {
        var (_, padding, featureShape, splitPct, layer) = Options;

        var result = TrainTest with { };
        
        if (layer > 0)
        {
            result = ApplyLayer(layer, result) with { };
        }

        if (featureShape is not null)
        {
            result = ApplyReshape(featureShape, result) with { };
        }

        if (padding > 0)
        {
            result = ApplyPadding(padding, result) with { };
        }

        if (splitPct > 0f)
        {
            result = ApplySplit(splitPct, result) with { };
        }

        return result;
    }

    private static TrainTestSplit ApplyLayer(int layer, TrainTestSplit trainTest)
    {
        var (trainFeatures, trainTargets, testFeatures, testTargets) = trainTest;
        var newFeatures = new List<Tensor>();
        var newTargets = new List<Tensor>();
        
        for (var i = 0; i < trainFeatures.Count - layer - 1; i++)
        {
            newFeatures.Add(trainFeatures.Skip(i).Take(layer).SelectMany(val => val.Read1D()).ToTensor());
            newTargets.Add(trainTargets.Skip(i + layer).Take(1).SelectMany(val => val.Read1D()).ToTensor());
        }

        return new TrainTestSplit(newFeatures, newTargets, testFeatures, testTargets);
    }

    private static TrainTestSplit ApplyReshape(Shape shape, TrainTestSplit trainTest) => 
        trainTest with { TrainFeatures = Reshape(trainTest.TrainFeatures, shape) };
    
    private static TrainTestSplit ApplyPadding(int padding, TrainTestSplit trainTest) =>
        trainTest with { TrainFeatures = trainTest.TrainFeatures.Select(row => row.Pad(padding)).ToList() };
    
    private static TrainTestSplit ApplySplit(float splitPct, TrainTestSplit trainTest)
    {
        var (trainFeatures, trainTargets, _, _) = trainTest;
        var splitIndex = (int) (trainFeatures.Count - (trainFeatures.Count * splitPct));

        return new TrainTestSplit(
            trainFeatures.Skip(splitIndex).ToList(), 
            trainTargets.Skip(splitIndex).ToList(), 
            trainFeatures.Take(splitIndex).ToList(),
            trainTargets.Take(splitIndex).ToList());
    }
    
    
    private static List<Tensor> Reshape(IEnumerable<Tensor> tensors, Shape shape)
    {
        var result = new Tensor[tensors.Count()];
        Parallel.For(0, result.Length, i => result[i] = tensors.ElementAt(i).Reshape(shape));
        return result.ToList();
    }
}