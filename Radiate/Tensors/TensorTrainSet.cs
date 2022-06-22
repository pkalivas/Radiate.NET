using Radiate.Extensions;
using Radiate.Records;
using Radiate.Tensors.Enums;
using Radiate.Tensors.Transforms;

namespace Radiate.Tensors;

public class TensorTrainSet
{
    private TrainTestSplit TrainTest { get; set; }
    private TensorTrainOptions Options { get; set; }
    private List<Batch> TrainBatchCache { get; set; } = new();
    private List<Batch> TestBatchCache { get; set; } = new();
    
    public TensorTrainSet(IEnumerable<float[]> features, IEnumerable<float[]> targets)
    {
        var inputs = features.Select(row => row.ToTensor()).ToList();
        var answers = targets.Select(row => row.ToTensor()).ToList();

        Options = new TensorTrainOptions();
        TrainTest = new TrainTestSplit(inputs, answers);
    }

    public TensorTrainSet(IEnumerable<float[]> features)
    {
        Options = new TensorTrainOptions();
        TrainTest = new TrainTestSplit(features.Select(row => row.ToTensor()).ToList(), new List<Tensor>());
    }

    public TensorTrainSet(Tensor features)
    {
        Options = new TensorTrainOptions();
        TrainTest = new TrainTestSplit(features.ToRows().ToList(), new List<Tensor>());
    }

    public TensorTrainSet(IEnumerable<Tensor> features, IEnumerable<Tensor> targets)
    {
        Options = new TensorTrainOptions();
        TrainTest = new TrainTestSplit(features.ToList(), targets.ToList());
    }

    public TensorTrainSet(TensorTrainOptions options)
    {
        Options = options;
    }

    public TensorTrainOptions TensorOptions => Options;

    public List<Batch> TrainingInputs => TrainBatchCache.Any() ? TrainBatchCache : TrainingBatches();

    public List<Batch> TestingInputs => TestBatchCache.Any() ? TestBatchCache : TestingBatches();

    public List<Batch> BatchAll => TrainingInputs.Concat(TestingInputs).ToList();

    public (List<float[]>, List<float[]>) RawTrainingInputs()
    {
        var shuffle = new ShuffleTransform();
        var split = new SplitTransform();
        var swapped = shuffle.Apply(TrainTest, Options, Enums.TrainTest.Train);
        var result = split.Apply(swapped.Item1, Options, Enums.TrainTest.Train);
        return (result.Item1.Features.Select(row => row.ToArray()).ToList(), result.Item1.Targets.Select(row => row.ToArray()).ToList());
    }

    public (List<float[]>, List<float[]>) RawTestingInputs()
    {
        var shuffle = new ShuffleTransform();
        var split = new SplitTransform();
        var swapped = shuffle.Apply(TrainTest, Options, Enums.TrainTest.Test);
        var result = split.Apply(swapped.Item1, Options, Enums.TrainTest.Test);
        return (result.Item1.Features.Select(row => row.ToArray()).ToList(), result.Item1.Targets.Select(row => row.ToArray()).ToList());
    }

    public List<(float[] feature, float[] target)> InputsToArrayRow(TrainTest trainTest = Enums.TrainTest.Train)
    {
        var result = new List<(float[], float[])>();
        var batches = trainTest switch
        {
            Enums.TrainTest.Train => TrainingInputs,
            Enums.TrainTest.Test => TestingInputs
        };

        foreach (var (feature, target) in batches)
        {
            result.AddRange(feature.Zip(target)
                .Select(pair => (pair.First.ToArray(), pair.Second.ToArray())));
        }

        return result;
    }
    
    public List<(Tensor feature, Tensor target)> InputsToTensorRow(TrainTest trainTest = Enums.TrainTest.Train)
    {
        var result = new List<(Tensor, Tensor)>();
        var batches = trainTest switch
        {
            Enums.TrainTest.Train => TrainingInputs,
            Enums.TrainTest.Test => TestingInputs
        };
        
        foreach (var (feature, target) in batches)
        {
            result.AddRange(feature.Zip(target)
                .Select(pair => (pair.First, pair.Second)));
        }

        return result;
    }

    public int OutputCategories => TrainTest.Targets
        .SelectMany(val => val)
        .Distinct()
        .Count();

    public float[] OutputCategoriesList => TrainTest.Targets
        .SelectMany(val => val)
        .Distinct()
        .ToArray();

    public Shape InputShape => TrainingInputs.First().Features.First().Shape;

    public Shape OutputShape => TrainingInputs.FirstOrDefault()?.Targets.FirstOrDefault()?.Shape;
    
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

    public TensorTrainSet Reshape(Shape shape)
    {
        Options = Options with { FeatureShape = shape };
        return this;
    }

    public TensorTrainSet Pad(int padding)
    {
        Options = Options with { Padding = padding };
        return this;
    }

    public TensorTrainSet Shuffle()
    {
        Options = Options with { Shuffle = true };
        return this;
    }

    public TensorTrainSet Kernel(FeatureKernel featureKernel, float c = 0f, float gamma = 0f)
    {
        Options = Options with { SpaceKernel = new SpaceKernel(featureKernel, c, gamma) };
        return this;
    }

    public TensorTrainSet TransformFeatures(Norm norm)
    {
        Options = Options with { FeatureNorm = norm };
        return this;
    }
    
    public TensorTrainSet TransformTargets(Norm norm)
    {
        Options = Options with { TargetNorm = norm };
        return this;
    }

    public TensorTrainSet Compile()
    {
        TrainingBatches();
        TestingBatches();
        return this;
    }

    public Tensor Process(Tensor input) => TensorSetTransforms.Process(input, Options);

    public Tensor Process(List<float[]> input)
    {
        var dataSet = input
            .Select(row => row.ToTensor())
            .ToArray();
        var stack = Tensor.Stack(dataSet, Axis.Zero);
        return TensorSetTransforms.Process(stack, Options);
    }
    
    private List<Batch> TrainingBatches()
    {
        var (trainTest, options) = TensorSetTransforms.Apply(TrainTest with { }, Options, Enums.TrainTest.Train);
        var batchSize = options.BatchSize;
        var trainBatches = new List<Batch>();

        for (var i = 0; i < trainTest.Features.Count; i += batchSize)
        {
            var batchFeatures = trainTest.Features
                .Skip(i)
                .Take(batchSize)
                .ToArray();
        
            var batchTargets = !trainTest.Targets.Any() ? Array.Empty<Tensor>() : trainTest.Targets
                .Skip(i)
                .Take(batchSize)
                .ToArray();
        
            trainBatches.Add(new Batch(batchFeatures, batchTargets));
        }

        TrainBatchCache = trainBatches;
        Options = options;
    
        return trainBatches;
    }

    private List<Batch> TestingBatches()
    {
        var (trainTest, options) = TensorSetTransforms.Apply(TrainTest with { }, Options, Enums.TrainTest.Test);
        var testBatches = new List<Batch>();
        for (var i = 0; i < trainTest.Features.Count; i++)
        {
            var batchFeatures = trainTest.Features
                .Skip(i)
                .Take(1)
                .ToArray();
            
            var batchTargets = !trainTest.Targets.Any() ? Array.Empty<Tensor>() :trainTest.Targets
                .Skip(i)
                .Take(1)
                .ToArray();
            
            testBatches.Add(new Batch(batchFeatures, batchTargets));
        }

        TestBatchCache = testBatches;
        Options = options;

        return testBatches;
    }
    
}