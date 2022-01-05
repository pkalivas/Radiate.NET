using Radiate.Domain.Extensions;
using Radiate.Domain.Records;

namespace Radiate.Domain.Tensors;

public static class TensorOperations
{
    public static Tensor Random(int height, int width = 0, int depth = 0)
    {
        var rand = new Random();
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
        var currentFlat = one.Flatten();
        var ten = Tensor.Like(otherShape);
        var count = 0;
        for (var j = 0; j < otherShape.Height; j++)
            if (otherShape.Width > 0)
                for (var k = 0; k < otherShape.Width; k++)
                    if (otherShape.Depth > 0)
                        for (var l = 0; l < otherShape.Depth; l++)
                            ten[j, k, l] = currentFlat[count++];
                    else
                        ten[j, k] = currentFlat[count++];
            else
                ten[j] = currentFlat[count++];

        return ten;
    }
    
    public static Tensor Fill(Shape shape, float value)
    {
        var result = shape switch
        {
            (> 0, <= 0, <= 0) => new float[shape.Height].ToTensor(),
            (> 0, > 0, <= 0) => new float[shape.Height, shape.Width].ToTensor(),
            (> 0, > 0, > 0) => new float[shape.Height, shape.Width, shape.Depth].ToTensor(),
            _ => throw new Exception($"Shape not supported")
        };
        
        
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
        result.Zero();
        
        for (var i = pad; i < height - pad; i++)
        for (var j = pad; j < width - pad; j++)
        for (var k = 0; k < depth; k++)
            result[i, j, k] = one[i - pad, j - pad, k];

        return result;
    }
    
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