using Radiate.Records;
using Radiate.Tensors.Enums;

namespace Radiate.Tensors.Transforms;

public class NormTransform : ITensorSetTransform
{
    public (TrainTestSplit, TensorTrainOptions) Apply(TrainTestSplit trainTest, TensorTrainOptions options, TrainTest train)
    {
        var (trainFeatures, trainTargets) = trainTest;

        if (options.FeatureNorm is not Norm.None)
        {
            var scalars = train is TrainTest.Test ? options.FeatureScalars : options.FeatureNorm switch
            {
                Norm.Normalize => GetNormalizeLookups(trainTest.Features),
                Norm.Standardize => GetStandardizeLookup(trainTest.Features),
                _ => options.FeatureScalars
            };

            options = options with { FeatureScalars = scalars };
            
            var newFeatures = options.FeatureNorm switch
            {
                Norm.Normalize => TensorOperations.Normalize(trainFeatures, options.FeatureScalars),
                Norm.Standardize => TensorOperations.Standardize(trainFeatures, options.FeatureScalars),
                Norm.OHE => TensorOperations.OneHotEncode(trainFeatures),
                Norm.Image => TensorOperations.ImageNormalize(trainFeatures),
                _ => throw new Exception($"Norm not implemented.")
            };

            trainTest = trainTest with { Features = newFeatures };
        }

        if (options.TargetNorm is not Norm.None)
        {
            var scalars = train is TrainTest.Test ? options.TargetScalars : options.TargetNorm switch
            {
                Norm.Normalize => GetNormalizeLookups(trainTest.Targets),
                Norm.Standardize => GetStandardizeLookup(trainTest.Targets),
                _ => options.TargetScalars
            };

            options = options with { TargetScalars = scalars };
            
            var newTargets = options.TargetNorm switch
            {
                Norm.Normalize => TensorOperations.Normalize(trainTargets, options.TargetScalars),
                Norm.Standardize => TensorOperations.Standardize(trainTargets, options.TargetScalars),
                Norm.OHE => TensorOperations.OneHotEncode(trainTargets),
                _ => throw new Exception($"Norm not implemented.")
            };

            trainTest = trainTest with { Targets = newTargets };
        }

        return (trainTest, options);
    }
    
    public Tensor Process(Tensor value, TensorTrainOptions options) => options.FeatureNorm switch
    {
        Norm.Image => TensorOperations.ImageNormalize(new List<Tensor> { value }).Single(),
        Norm.Normalize => TensorOperations.Normalize(new List<Tensor> { value }, options.FeatureScalars).Single(),
        Norm.Standardize => TensorOperations.Standardize(new List<Tensor> { value }, options.FeatureScalars).Single(),
        _ => value
    };
    
    private static NormalizeScalars GetNormalizeLookups(IReadOnlyCollection<Tensor> data)
    {
        var minLookup = new Dictionary<int, float>();
        var maxLookup = new Dictionary<int, float>();
        
        var featureLength = data
            .Select(row => row.Shape.Height)
            .Distinct()
            .Single();
        
        foreach (var index in Enumerable.Range(0, featureLength))
        {
            var column = data.Select(point => point[index]).ToList();
            minLookup[index] = column.Min();
            maxLookup[index] = column.Max();
        }

        return new NormalizeScalars(minLookup, maxLookup, new Dictionary<int, float>(), new Dictionary<int, float>());
    }
    
    private static NormalizeScalars GetStandardizeLookup(IReadOnlyCollection<Tensor> data)
    {
        var meanLookup = new Dictionary<int, float>();
        var stdLookup = new Dictionary<int, float>();
        
        var featureLength = data
            .Select(row => row.Shape.Height)
            .Distinct()
            .Single();
        
        foreach (var index in Enumerable.Range(0, featureLength))
        {
            var column = data.Select(point => point[index]).ToList();
            var average = column.Average();
            var sum = column.Sum(val => (float) Math.Pow(val - average, 2));
            
            meanLookup[index] = column.Average();
            stdLookup[index] = (float)Math.Sqrt(sum / (column.Count - 1));
        }

        return new NormalizeScalars(new Dictionary<int, float>(), new Dictionary<int, float>(), meanLookup, stdLookup);
    }
    
}