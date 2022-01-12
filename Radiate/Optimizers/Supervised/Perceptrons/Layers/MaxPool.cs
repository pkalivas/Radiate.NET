using Radiate.Gradients;
using Radiate.IO.Wraps;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons.Layers;

public class MaxPool : Layer
{
    private const float Tolerance = 1e-7f;
    
    private readonly Kernel _kernel;
    private readonly Stack<Tensor> _inputs;
    private readonly Stack<List<Slice>> _slices;
    private readonly SliceGenerator _sliceGenerator;
    private readonly int _stride;

    public MaxPool(MaxPoolWrap wrap) : base(wrap.Shape)
    {
        _kernel = wrap.Kernel;
        _inputs = new Stack<Tensor>();
        _slices = new Stack<List<Slice>>();
        _stride = wrap.Stride;
        _sliceGenerator = new SliceGenerator(_kernel, Shape.Depth, _stride);
    }
    
    public MaxPool(Shape shape, Kernel kernel, int stride) : base(shape)
    {
        _kernel = kernel;
        _inputs = new Stack<Tensor>();
        _slices = new Stack<List<Slice>>();
        _stride = stride;
        _sliceGenerator = new SliceGenerator(kernel, shape.Depth, stride);
    }

    public override Tensor Predict(Tensor input)
    {
        var slices = _sliceGenerator.Slice3D(input).ToList();
        return Pool(input, slices);
    }

    public override Tensor FeedForward(Tensor  input)
    {
        var slices = _sliceGenerator.Slice3D(input).ToList();
        var pooledResult = Pool(input, slices);
        
        _inputs.Push(input);
        _slices.Push(slices);
        
        return pooledResult;
    }

    public override Tensor PassBackward(Tensor  errors)
    {
        var prevInput = _inputs.Pop();
        var output = Tensor.Fill(prevInput.Shape, 0f);
        var previousSlices = _slices.Pop();

        Parallel.ForEach(previousSlices, pSlice =>
        {
            var (slice, sHeight, sWidth, sDepth) = pSlice;
            var (height, width, depth) = slice.Shape;
            var sliceMax = slice.Max();

            for (var i = 0; i < height; i++)
            {
                for (var j = 0; j < width; j++)
                {
                    for (var k = 0; k < depth; k++)
                    {
                        if (Math.Abs(slice[i, j, k] - sliceMax) < Tolerance)
                        {
                            var deltaHeight = sHeight * _stride + i;
                            var deltaWidth = sWidth * _stride + j;

                            output[deltaHeight, deltaWidth, sDepth] = errors[sHeight, sWidth, sDepth];
                        }
                    }
                }
            }
        });

        return output;
    }

    public override void UpdateWeights(GradientInfo gradient, int epoch, int batchSize) { }
    
    public override LayerWrap Save() => new()
    {
        LayerType = LayerType.MaxPool,
        MaxPool = new MaxPoolWrap
        {
            Shape = Shape,
            Kernel = _kernel,
            Stride = _stride
        }
    };

    private Tensor Pool(Tensor input, List<Slice> slices)
    {
        var (hStride, wStride) = _sliceGenerator.CalcStride(input);
        var output = new Tensor(hStride, wStride, Shape.Depth);

        Parallel.ForEach(slices, inSlice =>
        {
            var (slice, sHeight, sWidth, sDepth) = inSlice;
            output[sHeight, sWidth, sDepth] = slice.Max();
        });
        
        return output;
    }
    
}
