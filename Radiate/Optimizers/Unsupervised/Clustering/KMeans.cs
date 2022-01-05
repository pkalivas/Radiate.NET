using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Unsupervised.Clustering;

public class KMeans : IUnsupervised
{
    private readonly Random _random = new();

    private readonly int _kClusters;
    private readonly List<int>[] _clusters;
    private readonly Tensor[] _centroids;
    
    public KMeans(int kClusters)
    {
        _kClusters = kClusters;
        _clusters = new List<int>[kClusters];
        _centroids = new Tensor[kClusters];
    }
    
    public async Task Train(Batch batches, Func<Epoch, bool> trainFunc)
    {
        var (inputs, answers) = batches;
        
        foreach (var group in answers.GroupBy(val => Convert.ToInt32(val.Max())))
        {
            var idx = _random.Next(0, inputs.Length);
            _centroids[group.Key] = inputs[idx];
            _clusters[group.Key] = new List<int>();
        }
        
        var epochCount = 1;
        while (true)
        {
            var predictions = CreateClusters(inputs, answers);
            var newCentroids = UpdateCenters(inputs);

            for (var i = 0; i < _kClusters; i++)
            {
                _centroids[i] = newCentroids[i];
            }

            var loss = await CalcLoss(newCentroids);
            
            var classAcc = Validator.ClassificationAccuracy(predictions);
            var regAcc = Validator.RegressionAccuracy(predictions);
            var epoch = new Epoch(epochCount++, loss, classAcc, regAcc);
            
            if (trainFunc(epoch))
            {
                break;
            }
        }
    }

    public Prediction Predict(Tensor input)
    {
        var label = ClosestCenter(input);

        return new Prediction(new float[] { label }, label);
    }

    private List<(float[] output, float[] target)> CreateClusters(IReadOnlyList<Tensor> inputs, IReadOnlyList<Tensor> answers)
    {
        foreach (var cluster in _clusters)
        {
            cluster.Clear();
        }

        var predictions = new List<(float[] output, float[] target)>();
        for (var i = 0; i < inputs.Count; i++)
        {
            var input = inputs[i];
            var target = answers[i];
            var centerIdx = ClosestCenter(input);
            _clusters[centerIdx].Add(i);
                
            predictions.Add((new float[]{ centerIdx }, target.Read1D()));
        }

        return predictions;
    }

    private Tensor[] UpdateCenters(IReadOnlyList<Tensor> inputs)
    {
        var newCentroids = new Tensor[_kClusters];
        for (var i = 0; i < _kClusters; i++)
        {
            var newCenter = Tensor.Like(_centroids[i].Shape);

            foreach (var row in _clusters[i].Select(idx => inputs[idx]))
            {
                newCenter.Add(row);
            }

            newCentroids[i] = newCenter / _clusters[i].Count;
        }

        return newCentroids;
    }
    
    private int ClosestCenter(Tensor row)
    {
        var distances = _centroids
            .Select(point => EuclideanDistance(row, point))
            .ToList();

       return distances.IndexOf(distances.Min());
    }

    private async Task<float> CalcLoss(Tensor[] newCentroids) =>
        (await Task.WhenAll(_centroids.Zip(newCentroids)
            .Select(pair => Task.Run(() => EuclideanDistance(pair.First, pair.Second)))))
        .Sum();
    
    private static float EuclideanDistance(Tensor one, Tensor other)
    {
        var diff = (float) Math.Pow((one - other).Sum(), 2);
        return (float)Math.Sqrt(diff);
    }
    
}