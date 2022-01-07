using Radiate.Domain.Extensions;
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

    public KMeans(UnsupervisedWrap wrap)
    {
        _kClusters = wrap.KMeansWrap.KClusters;
        _clusters = wrap.KMeansWrap.Clusters;
        _centroids = wrap.KMeansWrap.Centroids;
    }

    public float Step(Tensor[] data, int epochCount)
    {
        if (epochCount == 0)
        {
            for (var i = 0; i < _kClusters; i++)
            {
                var idx = _random.Next(0, data.Length);
                _centroids[i] = data[idx];
                _clusters[i] = new List<int>();
            }
        }

        for (var i = 0; i < data.Length; i++)
        {
            var centerIdx = ClosestCenter(data[i]);
            _clusters[centerIdx].Add(i);
        }
            
        var newCentroids = UpdateCenters(data);

        for (var i = 0; i < _kClusters; i++)
        {
            _centroids[i] = newCentroids[i];
        }

        return CalcLoss(newCentroids);
    }

    public void Update()
    {
        foreach (var cluster in _clusters)
        {
            cluster.Clear();
        }
    }

    public Prediction Predict(Tensor input)
    {
        var label = ClosestCenter(input);
        var result = new float[] { label }.ToTensor();

        return new Prediction(result, label);
    }

    public UnsupervisedWrap Save() => new()
    {
        UnsupervisedType = UnsupervisedType.KMeans,
        KMeansWrap = new()
        {
            KClusters = _kClusters,
            Clusters = _clusters,
            Centroids = _centroids
        }
    };

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

    private float CalcLoss(Tensor[] newCentroids) =>
        _centroids.Zip(newCentroids)
            .Select(pair => EuclideanDistance(pair.First, pair.Second))
            .Sum();
    
    private static float EuclideanDistance(Tensor one, Tensor other)
    {
        var diff = (float) Math.Pow((one - other).Sum(), 2);
        return (float)Math.Sqrt(diff);
    }
    
}