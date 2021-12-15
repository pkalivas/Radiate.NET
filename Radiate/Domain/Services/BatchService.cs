using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Services;

public static class BatchService
{
    public static List<Batch> CreateBatches(
        IReadOnlyCollection<float[]> features,
        IReadOnlyCollection<float[]> targets, 
        Shape shape,
        int batchSize)
    {
        var batches = new List<Batch>();
        for (var i = 0; i < features.Count; i += batchSize)
        {
            var batchFeatures = features
                .Skip(i)
                .Take(batchSize)
                .Select(row => Transform(row, shape))
                .ToList();
            
            var batchTargets = targets
                .Skip(i)
                .Take(batchSize)
                .Select(row => new Tensor(row))
                .ToList();
            
            batches.Add(new(batchFeatures, batchTargets));
        }

        return batches;
    }

    public static Tensor Transform(float[] row, Shape shape) => shape switch
    {
        (> 0, > 0, > 0) => Transform3D(row, shape),
        (> 0, > 0, <=0 ) => Transform2D(row, shape),
        (> 0, <=0, <= 0) => new Tensor(row),
        (<= 0, <=0, <= 0) => new Tensor(row),
        _ => throw new Exception($"{shape} is not supported")
    };

    private static Tensor Transform2D(float[] row, Shape shape)
    {
        var (height, width, _) = shape;
        var result = new float[height, width];
        var count = 0;
        for (var i = 0; i < height; i++)
        {
            for (var j = 0; j < width; j++)
            {
                result[i, j] = row[count++];
            }
        }
        
        return new Tensor(result);
    }

    private static Tensor Transform3D(float[] row, Shape shape)
    {
        var (height, width, depth) = shape;
        var result = new float[height, width, depth];
        var count = 0;
        for (var i = 0; i < height; i++)
        {
            for (var j = 0; j < width; j++)
            {
                for (var k = 0; k < depth; k++)
                {
                    result[i, j, k] = row[count++];
                }
            }
        }
        
        return new Tensor(result);
    }
}