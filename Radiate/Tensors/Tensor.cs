using System.Collections;
using Newtonsoft.Json;
using Radiate.Extensions;
using Radiate.Records;
using Radiate.Tensors.Enums;

namespace Radiate.Tensors;

[JsonObject]
public class Tensor : IEnumerable<float>
{
    [JsonProperty]
    private float[] ElementsOneD { get; set; }
    
    [JsonProperty]
    private float[,] ElementsTwoD { get; set; }

    [JsonProperty]
    private float[,,] ElementsThreeD { get; set; }
    
    [JsonProperty]
    // ReSharper disable once InconsistentNaming
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

        _shape = new Shape(height, width, depth);
    }
    
    public Tensor(float[] elements)
    {
        ElementsOneD = elements;
        _shape = new Shape(elements.Length);
    }

    public Tensor(float[,] elements)
    {
        ElementsTwoD = elements;
        _shape = new Shape(elements.GetLength(0), elements.GetLength(1));

    }

    public Tensor(float[,,] elements)
    {
        ElementsThreeD = elements;
        _shape = new Shape(elements.GetLength(0), elements.GetLength(1), elements.GetLength(2));

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

    public Tensor[] ToRows()
    {
        var (height, _, _) = Shape;
        var result = new Tensor[height];
        for (var i = 0; i < height; i++)
        {
            result[i] = Row(i);
        }

        return result;
    }

    public Tensor Row(int index)
    {
        if (Shape.Width == 0)
        {
            throw new Exception("Cannot take row from 1D Tensor.");
        }

        var result = new Tensor(Shape.Width);
        for (var i = 0; i < Shape.Width; i++)
        {
            result[i] = this[index, i];
        }

        return result;
    }

    public Tensor Column(int index)
    {
        if (Shape.Width == 0)
        {
            throw new Exception("Cannot take column from 1D Tensor.");
        }

        var result = new Tensor(Shape.Height);
        for (var i = 0; i < Shape.Height; i++)
        {
            result[i] = this[i, index];
        }

        return result;
    }

    public Tensor Plane(int index)
    {
        if (Shape.Depth == 0)
        {
            throw new Exception("Cannot take Plane from 1D or 2D Tensor.");
        }

        var result = new Tensor(Shape.Height, Shape.Width, 1);
        for (var i = 0; i < Shape.Height; i++)
        {
            for (var j = 0; j < Shape.Width; j++)
            {
                result[i, j, 0] = this[i, j, index];
            }
        }

        return result;
    }

    public Tensor Flatten() => this.ToArray().ToTensor();

    public Tensor Reshape(Shape otherShape) =>
        TensorOperations.Reshape(this, otherShape);
    
    public Tensor Slice(int[] height, int[] width, int[] depth) =>
        TensorOperations.Slice(this, height, width, depth);

    public Tensor Pad(int pad) =>
        TensorOperations.Pad(this, pad);

    public Tensor Diag() =>
        TensorOperations.Diagonal(this);

    public Tensor Tile() =>
        TensorOperations.Tile(this);

    public Tensor T() =>
        TensorOperations.Transpose(this);

    public Tensor Sign() =>
        TensorOperations.Sign(this);
    
    public Tensor RadialBasis(float gamma) =>
        TensorOperations.RadialBasis(this, gamma);

    public Tensor Polynomial(float dim) =>
        TensorOperations.Polynomial(this, dim);

    public Tensor Unique() => this.Distinct().ToTensor();

    public float Max() => TensorMath.Max(this);
    
    public float Entropy() => TensorMath.Entropy(this);

    public float PlaneDot(Tensor other, int index) => 
        TensorMath.PlaneDot(this, other, index);
    
    public void Add(Tensor other) => this.AddInPlace(other);

    public void Subtract(Tensor other) => this.SubtractInPlace(other);

    public void Multiply(Tensor other) => this.MultiplyInPlace(other);
    
    public void Mul(float value) => this.MultiplyInPlace(value);
    
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

    public int MaxIdx()
    {
        var idx = 0;
        var best = float.MinValue;
        for (var i = 0; i < Shape.Height; i++)
        {
            if (this[i] > best)
            {
                idx = i;
                best = this[i];
            }
        }

        return idx;
    }
    
    public int MinIdx()
    {
        var idx = 0;
        var best = float.MaxValue;
        for (var i = 0; i < Shape.Height; i++)
        {
            if (this[i] < best)
            {
                idx = i;
                best = this[i];
            }
        }

        return idx;
    }

    public static float Dot(Tensor one, Tensor two) => TensorMath.Dot(one, two);

    public static Tensor Stack(Tensor[] tensors, Axis axis) => axis switch
    {
        Axis.Zero => tensors.Skip(1).Aggregate(tensors[0], Stacker.StackZero),
        Axis.One => tensors.Skip(1).Aggregate(tensors[0], Stacker.StackOne),
        _ => throw new Exception("Cannot stack tensors.")
    };

    public static Tensor Like(Tensor other) => Like(other.Shape);
    
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

    public static Tensor Fill(Shape shape, float value) =>
        TensorOperations.Fill(shape, value);

    public static Tensor Random(int height, int width = 0, int depth = 0) =>
        TensorOperations.Random(height, width, depth);
    
    public static Tensor Apply(Tensor ten, Func<float, float> func) => 
        TensorOperations.Apply(ten, func);
    
    private static Tensor Apply(Tensor one, Tensor two, Func<float, float, float> func) =>
        TensorOperations.Apply(one, two, func);

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

    public IEnumerator<float> GetEnumerator()
    {
        var (height, width, depth) = Shape;
        var dim = GetDimension();

        if (dim == 1)
        {
            for (var i = 0; i < height; i++)
            {
                yield return this[i];
            }
        }

        if (dim == 2)
        {
            for (var i = 0; i < height; i++)
            {
                for (var j = 0; j < width; j++)
                {
                    yield return this[i, j];
                }
            }
        }

        if (dim == 3)
        {
            for (var i = 0; i < height; i++)
            {
                for (var j = 0; j < width; j++)
                {
                    for (var k = 0; k < depth; k++)
                    {
                        yield return this[i, j, k];
                    }
                }
            }
        }
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}