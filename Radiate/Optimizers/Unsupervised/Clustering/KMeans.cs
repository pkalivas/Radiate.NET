using Radiate.Domain.Loss;
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
    
    public async Task Train(Batch batches, LossFunction lossFunction, Func<Epoch, bool> trainFunc)
    {
        var (inputs, answers) = batches;
        
        foreach (var group in answers.GroupBy(val => Convert.ToInt32(val.Max())))
        {
            var idx = _random.Next(0, inputs.Count);
            _centroids[group.Key] = inputs[idx];
            _clusters[group.Key] = new List<int>();
        }
        
        var epochCount = 1;
        while (true)
        {
            var predictions = CreateClusters(inputs, answers, lossFunction);
            var newCentroids = UpdateCenters(inputs);

            for (var i = 0; i < _kClusters; i++)
            {
                _centroids[i] = newCentroids[i];
            }

            var loss = await CalcLoss(newCentroids, lossFunction);
            
            var (_, classAcc, regAcc) = Validator.ValidateEpoch(new List<float>(), predictions);
            var epoch = new Epoch(epochCount++, loss, classAcc, regAcc);
            
            if (trainFunc(epoch))
            {
                break;
            }
        }
    }

    public Prediction Predict(Tensor input, LossFunction lossFunction)
    {
        var label = ClosestCenter(input, lossFunction);

        return new Prediction(new float[] { label }, label);
    }

    private List<(float[] output, float[] target)> CreateClusters(
        IReadOnlyList<Tensor> inputs, 
        IReadOnlyList<Tensor> answers,
        LossFunction lossFunction)
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
            var centerIdx = ClosestCenter(input, lossFunction);
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
    
    private int ClosestCenter(Tensor row, LossFunction lossFunction)
    {
        var distances = _centroids
            .Select(point => lossFunction(row, point).Loss)
            .ToList();

       return distances.IndexOf(distances.Min());
    }

    private async Task<float> CalcLoss(Tensor[] newCentroids, LossFunction lossFunction) =>
        (await Task.WhenAll(_centroids.Zip(newCentroids)
            .Select(pair => Task.Run(() => lossFunction(pair.First, pair.Second).Loss))))
        .Sum();
    
}