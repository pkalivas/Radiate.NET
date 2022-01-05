﻿using Newtonsoft.Json;
using Radiate.Domain.Records;

namespace Radiate.Domain.Tensors;

public class Tensor
{
    [JsonProperty]
    private float[] ElementsOneD { get; set; }
    
    [JsonProperty]
    private float[,] ElementsTwoD { get; set; }

    [JsonProperty]
    private float[,,] ElementsThreeD { get; set; }
    
    [JsonProperty]
    private Shape _shape { get; set; }
    
    public Tensor() { }

    public Tensor(int height, int width = 0, int depth = 0)
    {
        switch (height)
        {
            case > 0 when width > 0 && depth > 0:
                ElementsThreeD = new float[height, width, depth];
                break;
            case > 0 when width > 0 && depth <= 0:
                ElementsTwoD = new float[height, width];
                break;
            case > 0 when width <= 0 && depth <= 0:
                ElementsOneD = new float[height];
                break;
            default:
                throw new Exception($"Cannot make Tensor with {height} {width} {depth}");
        }

        _shape = SetShape();
    }
    
    public Tensor(float[] elements)
    {
        ElementsOneD = elements;
        _shape = SetShape();
    }

    public Tensor(float[,] elements)
    {
        ElementsTwoD = elements;
        _shape = SetShape();

    }

    public Tensor(float[,,] elements)
    {
        ElementsThreeD = elements;
        _shape = SetShape();

    }
    
    public Shape Shape => _shape;
    

    public float this[int i]
    {
        get => ElementsOneD[i];
        set => ElementsOneD[i] = value;
    }

    public float this[int i, int j]
    {
        get => ElementsTwoD[i, j];
        set => ElementsTwoD[i, j] = value;
    }

    public float this[int i, int j, int k]
    {
        get => ElementsThreeD[i, j, k];
        set => ElementsThreeD[i, j, k] = value;
    }
    
    private Shape SetShape() => GetDimension() switch
    {
        1 => new Shape(ElementsOneD.Length, 0, 0),
        2 => new Shape(ElementsTwoD.GetLength(0), ElementsTwoD.GetLength(1), 0),
        3 => new Shape(ElementsThreeD.GetLength(0), ElementsThreeD.GetLength(1), ElementsThreeD.GetLength(2)),
        _ => new Shape(0, 0, 0),
    };

    public float[] Read1D() => ElementsOneD;

    public float[,] Read2D() => ElementsTwoD;

    public float[,,] Read3D() => ElementsThreeD;


    
    public static float Sum(Tensor one, Tensor other)
    {
        var oShape = one.Shape;
        var tShape = other.Shape;
        
        if (oShape.Width != tShape.Width || oShape.Height != tShape.Height || oShape.Depth !=  tShape.Depth)
        {
            throw new Exception($"Cannot dot product two unlike matrices");
        }

        var result = 0.0f;
        for (var i = 0; i < oShape.Height; i++) 
        {
            for (var j = 0; j < oShape.Width; j++) 
            {
                for (var k = 0; k < oShape.Depth; k++) 
                {
                    result += one.ElementsThreeD[i, j, k] * other.ElementsThreeD[i, j, k];
                }
            }
        }

        return result;
    }

    public float Sum()
    {
        var (height, width, depth) = Shape;
        var total = 0f;
        var dimension = GetDimension();
        
        if (dimension == 1)
        {
            total = Read1D().Sum();
        }

        if (dimension == 2)
        {
            for (var i = 0; i < height; i++)
            {
                for (var j = 0; j < width; j++)
                {
                    total += this[i, j];
                }
            }
        }

        if (dimension == 3)
        {
            for (var i = 0; i < height; i++)
            {
                for (var j = 0; j < width; j++)
                {
                    for (var k = 0; k < depth; k++)
                    {
                        total += this[i, j, k];
                    }
                }
            } 
        }

        return total;
    }

    public Tensor Flatten()
    {
        var newInput = new List<float>();

        for (var j = 0; j < _shape.Height; j++)
            if (_shape.Width > 0)
                for (var k = 0; k < _shape.Width; k++)
                    if (_shape.Depth > 0)
                        for (var l = 0; l < _shape.Depth; l++)
                            newInput.Add(this[j, k, l]);
                    else
                        newInput.Add(this[j, k]);
            else
                newInput.Add(this[j]);

        return new Tensor(newInput.ToArray());
    }

    public Tensor Reshape(Shape otherShape)
    {
        var currentFlat = Flatten();
        var ten = Like(otherShape);
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

    public Tensor Slice(int[] height, int[] width, int[] depth)
    {
        var sliceH = height[1] - height[0];
        var sliceW = width[1] - width[0];
        var sliceD = depth[1] - depth[0];
        var slice = new float[sliceH, sliceW, sliceD];

        if (ElementsThreeD is null)
        {
            throw new Exception($"No Tensor to slice");
        }
        
        if (height[1] < height[0] || width[1] < width[0] || depth[1] < depth[0])
        {
            throw new Exception("Malformed pairs.");
        }

        if (height[1] > Shape.Height || width[1] > Shape.Width || depth[1] > Shape.Depth)
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

                    slice[i, j, k] = ElementsThreeD[coordI, coordJ, coordK];
                }
            }
        }    
        
        return new Tensor(slice);
    }

    public Tensor Pad(int pad)
    {
        var result = new float[pad * 2 + Shape.Height, pad * 2 + Shape.Width, Shape.Depth].ToTensor();
        
        result.Zero();
        
        for (var i = pad; i < result._shape.Height - pad; i++)
            for (var j = pad; j < result._shape.Width - pad; j++)
            for (var k = 0; k < result._shape.Depth; k++)
                result[i, j, k] = this[i - pad, j - pad, k];

        return result;
    }

    public float Max()
    {
        var (height, width, depth) = Shape;
        var bestVal = float.MinValue;
        var dimension = GetDimension();
        
        if (dimension == 1)
        {
            bestVal = Read1D().Max();
        }

        if (dimension == 2)
        {
            for (var i = 0; i < height; i++)
            {
                for (var j = 0; j < width; j++)
                {
                    if (this[i, j] > bestVal)
                    {
                        bestVal = this[i, j];
                    }
                }
            }
        }

        if (dimension == 3)
        {
            for (var i = 0; i < height; i++)
            {
                for (var j = 0; j < width; j++)
                {
                    for (var k = 0; k < depth; k++)
                    {
                        if (this[i, j, k] > bestVal)
                        {
                            bestVal = this[i, j, k];
                        }
                    }
                }
            } 
        }

        return bestVal > float.MinValue ? bestVal : 0;
    }
    
    
    public static Tensor Like(Shape shape) => shape switch
    {
        (> 0, <= 0, <= 0) => new Tensor(new float[shape.Height]),
        (> 0, > 0, <= 0) => new Tensor(new float[shape.Height, shape.Width]),
        (> 0, > 0, > 0) => new Tensor(new float[shape.Height, shape.Width, shape.Depth]),
        _ => throw new Exception($"Shape not supported")
    };

    public static Tensor ARange(int range) => Enumerable
        .Range(0, range)
        .Select(val => (float)val)
        .ToArray()
        .ToTensor();
    

    public void Zero()
    {
        for (var j = 0; j < _shape.Height; j++)
            if (_shape.Width > 0)
                for (var k = 0; k < _shape.Width; k++)
                    if (_shape.Depth > 0)
                        for (var l = 0; l < _shape.Depth; l++)
                            this[j, k, l] = 0f;
                    else
                        this[j, k] = 0f;
            else
                this[j] = 0f;
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
    
    public void Add(Tensor other)
    {
        for (var j = 0; j < _shape.Height; j++)
            if (_shape.Width > 0)
                for (var k = 0; k < _shape.Width; k++)
                    if (_shape.Depth > 0)
                        for (var l = 0; l < _shape.Depth; l++)
                            ElementsThreeD[j, k, l] += other.ElementsThreeD[j, k, l];
                    else
                        ElementsTwoD[j, k] += other.ElementsTwoD[j, k];
            else
                ElementsOneD[j] += other.ElementsOneD[j];
    }

    public void Mul(float value)
    {
        for (var j = 0; j < _shape.Height; j++)
            if (_shape.Width > 0)
                for (var k = 0; k < _shape.Width; k++)
                    if (_shape.Depth > 0)
                        for (var l = 0; l < _shape.Depth; l++)
                            ElementsThreeD[j, k, l] *= value;
                    else
                        ElementsTwoD[j, k] *= value;
            else
                ElementsOneD[j] *= value;
    }
    
    public static Tensor Random(int height, int width = 0, int depth = 0)
    {
        var rand = new Random();
        var result = new Tensor(height, width, depth);
        var dimension = result.GetDimension();

        if (dimension == 1)
        {
            TensorLoop((i) => result[i] = (float) rand.NextDouble() * 2 - 1, height);
        }

        if (dimension == 2)
        {
            TensorLoop((i, j) =>
            {
                result[i, j] = (float)rand.NextDouble() * 2 - 1;
            }, height, width);
        }

        if (dimension == 3)
        {
            TensorLoop((i, j, k) =>
            {
                result[i, j, k] = (float)rand.NextDouble() * 2 - 1;
            }, height, width, depth);
        }
        
        return result;
    }

    public string ToImageString(float scale = .5f)
    {
        var shape = Shape;
        var result = "";
        for (var i = 0; i < shape.Height; i++)
        {
            for (var j = 0; j < shape.Width; j++)
            {
                for (var k = 0; k < shape.Depth; k++)
                {
                    var value = ElementsThreeD[i, j, k];
                    result += value > scale ? "*" : " ";
                }
            }

            result += "\n";
        }

        return result;
    }

    public static Tensor Apply(Tensor ten, Func<float, float> func)
    {
        var result = Like(ten.Shape);
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
    
    private static Tensor Apply(Tensor one, Tensor two, Func<float, float, float> func)
    {
        if (one.Shape != two.Shape)
        {
            throw new Exception("Element wise operations must have matching shapes");
        }

        var result = Like(one.Shape);
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


    public static void TensorLoop(Action<int, int, int> act, int height, int width, int depth)
    {
        for (var i = 0; i < height; i++)
        {
            for (var j = 0; j < width; j++)
            {
                for (var k = 0; k < depth; k++)
                {
                    act(i, j, k);
                }    
            }
        }
    }

    public static void TensorLoop(Action<int, int> act, int height, int width)
    {
        for (var i = 0; i < height; i++)
        {
            for (var j = 0; j < width; j++)
            {
                act(i, j);
            }
        }
    }

    public static void TensorLoop(Action<int> act, int height)
    {
        for (var i = 0; i < height; i++)
        {
            act(i);
        }
    }



    private void SetValue(float value, int i, int j = 0, int k = 0)
    {
        var dimension = GetDimension();

        switch (dimension)
        {
            case 1:
                ElementsOneD[i] = value;
                break;
            case 2:
                ElementsTwoD[i, j] = value;
                break;
            case 3:
                ElementsThreeD[i, j, k] = value;
                break;
        }
    }

    public int GetDimension() => _shape switch
    {
        (> 0, <= 0, <= 0) => 1,
        (> 0, > 0, <= 0) => 2,
        (> 0, > 0, > 0) => 3,
        _ => throw new Exception($"Dimension not supported")
    };
    
    
    
    public static Tensor operator +(Tensor one, Tensor two) => Apply(one, two, (i, j) => i + j);
    public static Tensor operator +(Tensor one, float num) => Apply(one, val => val + num);
    public static Tensor operator +(float num, Tensor one) => Apply(one, val => val + num);

    
    public static Tensor operator *(Tensor one, Tensor two) => Apply(one, two, (i, j) => i * j);
    public static Tensor operator *(Tensor one, float num) => Apply(one, val => val * num);
    public static Tensor operator *(float num, Tensor one) => Apply(one, val => num * val);
    
    public static Tensor operator -(Tensor one, Tensor two) => Apply(one, two, (i, j) => i - j);
    public static Tensor operator -(Tensor one, float num) => Apply(one, val => val - num);
    public static Tensor operator -(float num, Tensor one) => Apply(one, val => num - val);

    
    public static Tensor operator /(Tensor one, Tensor two) => Apply(one, two, (i, j) => i / j);
    public static Tensor operator /(Tensor one, float num) => Apply(one, val => val / num);
    public static Tensor operator /(float num, Tensor one) => Apply(one, val => num / val);
    
}

public static class ArrayExtensions
{
    public static Tensor ToTensor(this float[] arr) => new(arr);

    public static Tensor ToTensor(this float[,] arr) => new(arr);

    public static Tensor ToTensor(this float[,,] arr) => new(arr);

    public static Tensor ToTensor(this IEnumerable<float> en) => en.ToArray().ToTensor();
}
