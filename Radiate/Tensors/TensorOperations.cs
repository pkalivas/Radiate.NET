using Radiate.Extensions;
using Radiate.Records;

namespace Radiate.Tensors;

public static class TensorOperations
{
    private const float Tolerance = 0.001f;

    public static Tensor Diagonal(Tensor values)
    {
        var result = new Tensor(values.Count(), values.Count());
        for (var i = 0; i < result.Shape.Height; i++)
        {
            for (var j = 0; j < result.Shape.Width; j++)
            {
                result[i, j] = j == i ? values[i] : 0;
            }
        }

        return result;
    }

    public static Tensor Tile(Tensor values)
    {
        var result = new Tensor(values.Count(), values.Count());
        for (var i = 0; i < result.Shape.Height; i++)
        {
            for (var j = 0; j < result.Shape.Width; j++)
            {
                result[i, j] = values[i];
            }
        }

        return result;
    }

    public static Tensor Transpose(Tensor values)
    {
        var dim = values.GetDimension();
        var result = new Tensor(values.Shape.Height, values.Shape.Height);
        
        if (dim == 1)
        {
            for (var i = 0; i < result.Shape.Height; i++)
            {
                for (var j = 0; j < result.Shape.Width; j++)
                {
                    result[i, j] = values[j];
                }
            }
        }

        if (dim == 2)
        {
            for (var i = 0; i < result.Shape.Height; i++)
            {
                for (var j = 0; j < result.Shape.Width; j++)
                {
                    result[i, j] = values[j, i];
                }
            }
        }

        if (dim == 3)
        {
            throw new Exception("Cannot transpose 3D Tensor");
        }

        return result;
    }
    
    public static Tensor Random(int height, int width = 0, int depth = 0)
    {
        var rand = RandomGenerator.RandomGenerator.Next;
        var result = new Tensor(height, width, depth);
        var dimension = result.GetDimension();

        if (dimension == 1)
        {
            Tensor.TensorLoop((i) => result[i] = (float) rand.NextDouble() * 2 - 1, height);
        }

        if (dimension == 2)
        {
            Tensor.TensorLoop((i, j) =>
            {
                result[i, j] = (float)rand.NextDouble() * 2 - 1;
            }, height, width);
        }

        if (dimension == 3)
        {
            Tensor.TensorLoop((i, j, k) =>
            {
                result[i, j, k] = (float)rand.NextDouble() * 2 - 1;
            }, height, width, depth);
        }
        
        return result;
    }
    
    public static Tensor Reshape(Tensor one, Shape otherShape)
    {
        var ten = Tensor.Like(otherShape);
        var tDim = ten.GetDimension();

        if (tDim == 1)
        {
            return one.Flatten();
        }

        if (tDim == 2)
        {
            using var oneEnumerator = one.GetEnumerator();
            for (var i = 0; i < otherShape.Height; i++)
            {
                for (var j = 0; j < otherShape.Width; j++)
                {
                    oneEnumerator.MoveNext(); 

                    ten[i, j] = oneEnumerator.Current;
                }
            }
        }

        if (tDim == 3)
        {
            using var oneEnumerator = one.GetEnumerator();
            for (var i = 0; i < otherShape.Height; i++)
            {
                for (var j = 0; j < otherShape.Width; j++)
                {
                    for (var k = 0; k < otherShape.Depth; k++)
                    {
                        oneEnumerator.MoveNext();   

                        ten[i, j, k] = oneEnumerator.Current;
                    }
                }
            }
        }
        
        return ten;
    }
    
    public static Tensor Fill(Shape shape, float value)
    {
        var result = Tensor.Like(shape);
        
        for (var j = 0; j < shape.Height; j++)
            if (shape.Width > 0)
                for (var k = 0; k < shape.Width; k++)
                    if (shape.Depth > 0) 
                        for (var l = 0; l < shape.Depth; l++)
                            result[j, k, l] = value;
                    else
                        result[j, k] = value;
            else
                result[j] = value;

        return result;
    }
    
    public static Tensor Pad(Tensor one, int pad)
    {
        var result = new float[pad * 2 + one.Shape.Height, pad * 2 + one.Shape.Width, one.Shape.Depth].ToTensor();
        var (height, width, depth) = result.Shape;
        
        for (var i = pad; i < height - pad; i++)
        for (var j = pad; j < width - pad; j++)
        for (var k = 0; k < depth; k++)
            result[i, j, k] = one[i - pad, j - pad, k];

        return result;
    }

    public static Tensor Sign(Tensor one)
    {
        var result = Tensor.Like(one);
        var (height, width, depth) = one.Shape;
        
        var func = (float val) => val < 0 ? -1 : val == 0 ? 0 : 1;
        
        for (var j = 0; j < height; j++)
            if (width > 0)
                for (var k = 0; k < width; k++)
                    if (depth > 0) 
                        for (var l = 0; l < depth; l++)
                            result[j, k, l] = func(one[j, k, l]);
                    else
                        result[j, k] = func(one[j, k]);
            else
                result[j] = func(one[j]);

        return result;
    }

    public static Tensor RadialBasis(Tensor one, float gamma)
    {
        var gVal = gamma <= 0 ? 1f / one.Count() : gamma;
        var newTensor = new Tensor(one.Shape.Height * one.Shape.Height);

        for (var i = 0; i < one.Shape.Height; i++)
        {
            for (var j = 0; j < one.Shape.Height; j++)
            {
                if (i == j)
                {
                    newTensor[i * one.Shape.Height + j] += one[j];
                }
                else
                {
                    var diff = (float)Math.Pow(one[i] - one[j], 2);

                    newTensor[i * one.Shape.Height + j] += (float)Math.Exp(-diff / (2f * (gVal * gVal)));    
                }
                   
            }
        }

        return newTensor;
    }

    public static Tensor Polynomial(Tensor one, float dim)
    {
        var newTensor = new Tensor(one.Shape.Height * one.Shape.Height);
        for (var i = 0; i < one.Shape.Height; i++)
        {
            for (var j = 0; j < one.Shape.Height; j++)
            {
                if (i == j)
                {
                    newTensor[i * one.Shape.Height + j] += one[j];
                }
                else
                {
                    var diff = one[i] * one[j];

                    newTensor[i * one.Shape.Height + j] += 1f + (float)Math.Pow(diff, dim);    
                }
                
            }
        }

        return newTensor;
    }
    
    public static List<Tensor> Normalize(IEnumerable<Tensor> data, NormalizeScalars scalars)
    {
        var (minLookup, maxLookup, _, _) = scalars;

        var result = new List<Tensor>();
        foreach (var feature in data)
        {
            if (feature.GetDimension() is 1)
            {
                result.Add(feature
                    .Select((val, index) =>
                    {
                        var min = minLookup[index];
                        var max = maxLookup[index];

                        var denominator = Math.Abs(min - max) < Tolerance ? 1 : max - min;
                        return (val / (min == 0 ? 1 : min)) / denominator;
                    })
                    .ToTensor());
            }

            if (feature.GetDimension() is 2)
            {
                var temp = Tensor.Like(feature.Shape);
                for (var i = 0; i < feature.Shape.Height; i++)
                {
                    for (var j = 0; j < feature.Shape.Width; j++)
                    {
                        var min = minLookup[j];
                        var max = maxLookup[j];

                        var denominator = Math.Abs(min - max) < Tolerance ? 1 : max - min;
                        var value = (feature[i, j] / (min == 0 ? 1 : min)) / denominator;
                        temp[i, j] = value;
                    }
                }
                
                result.Add(temp);
            }
        }

        return result;
    }
    
    public static List<Tensor> Standardize(IEnumerable<Tensor> data, NormalizeScalars scalars)
    {
        var (_, _, meanLookup, stdLookup) = scalars;
        var result = new List<Tensor>();
        foreach (var feature in data)
        {
            if (feature.GetDimension() is 1)
            {
                result.Add(feature
                    .Select((val, index) => (val - meanLookup[index]) / stdLookup[index])
                    .ToTensor());
            }

            if (feature.GetDimension() is 2)
            {
                var temp = Tensor.Like(feature);
                for (var i = 0; i < feature.Shape.Height; i++)
                {
                    for (var j = 0; j < feature.Shape.Width; j++)
                    {
                        var mean = meanLookup[j];
                        var std = stdLookup[j];
                        temp[i, j] = (feature[i, j] - mean) / std;
                    }
                }
                result.Add(temp);
            }
        }

        return result;
    }
    
    public static List<Tensor> OneHotEncode(List<Tensor> targets)
    {
        var targetCount = targets.SelectMany(row => row).Distinct().Count();
        return targets
            .Select(tar => Enumerable
                .Range(0, targetCount)
                .Select((_, index) => Math.Abs(index - tar.First()) < Tolerance ? 1f : 0.0f)
                .ToTensor())
            .ToList();
    }

    public static List<Tensor> ImageNormalize(IEnumerable<Tensor> features) => 
        features.Select(row => Tensor.Apply(row, val => val / 255f)).ToList();
    
    
    public static Tensor Slice(Tensor one, int[] height, int[] width, int[] depth)
    {
        var sliceH = height[1] - height[0];
        var sliceW = width[1] - width[0];
        var sliceD = depth[1] - depth[0];
        var slice = new float[sliceH, sliceW, sliceD];

        if (one.GetDimension() != 3)
        {
            throw new Exception($"No Tensor to slice");
        }
        
        if (height[1] < height[0] || width[1] < width[0] || depth[1] < depth[0])
        {
            throw new Exception("Malformed pairs.");
        }

        if (height[1] > one.Shape.Height || width[1] > one.Shape.Width || depth[1] > one.Shape.Depth)
        {
            throw new Exception("Slice exceeds dimensions.");
        }

        for (var i = 0; i < sliceH; i++)
        {
            for (var j = 0; j < sliceW; j++) 
            {
                for (var k = 0; k < sliceD ; k++)
                {
                    var coordI = height[0] + i;
                    var coordJ = width[0] + j;
                    var coordK = depth[0] + k;

                    slice[i, j, k] = one[coordI, coordJ, coordK];
                }
            }
        }    
        
        return new Tensor(slice);
    }
    
    public static Tensor Apply(Tensor ten, Func<float, float> func)
    {
        var result = Tensor.Like(ten.Shape);
        var (height, width, depth) = result.Shape;
        
        for (var j = 0; j < height; j++)
            if (width > 0)
                for (var k = 0; k < width; k++)
                    if (depth > 0)
                        for (var l = 0; l < depth; l++)
                            result[j, k, l] = func(ten[j, k, l]);
                    else
                        result[j, k] = func(ten[j, k]);
            else
                result[j] = func(ten[j]);
        
        return result;
    }

    public static Tensor Apply(Tensor one, Tensor two, Func<float, float, float> func)
    {
        if (one.Shape != two.Shape)
        {
            throw new Exception("Element wise operations must have matching shapes");
        }

        var result = Tensor.Like(one.Shape);
        var (height, width, depth) = result.Shape;

        for (var j = 0; j < height; j++)
            if (width > 0)
                for (var k = 0; k < width; k++)
                    if (depth > 0)
                        for (var l = 0; l < depth; l++)
                            result[j, k, l] = func(one[j, k, l], two[j, k, l]);
                    else
                        result[j, k] = func(one[j, k], two[j, k]);
            else
                result[j] = func(one[j], two[j]);
        
        return result;
    }
}