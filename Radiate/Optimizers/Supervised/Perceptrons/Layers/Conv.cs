using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons.Layers;

public class Conv : Layer
{
    private readonly IActivationFunction _activation;
    private readonly Kernel _kernel;
    private readonly SliceGenerator _sliceGenerator;
    private readonly Stack<List<Slice>> _slices;
    private readonly Stack<Tensor> _inputs;
    private readonly Stack<Tensor> _outputs;
    private readonly Tensor[] _filters;
    private readonly Tensor[] _filterGradients;
    private readonly Tensor _bias;
    private readonly Tensor _biasGradients;

    public Conv(ConvWrap wrap) : base(wrap.Shape)
    {
        _activation = ActivationFunctionFactory.Get(wrap.Activation);
        _kernel = wrap.Kernel;
        _sliceGenerator = new SliceGenerator(_kernel, wrap.Shape.Depth, wrap.Stride);
        _slices = new Stack<List<Slice>>();
        _inputs = new Stack<Tensor>();
        _outputs = new Stack<Tensor>();
        _filters = wrap.Filters;
        _filterGradients = wrap.FilterGradients;
        _bias = wrap.Bias;
        _biasGradients = wrap.BiasGradients;
    }
    
    public Conv(Shape shape, Kernel kernel, int stride, IActivationFunction activation) : base(shape)
    {
        var (count, dim) = kernel;
        var (_, _, depth) = shape;
        
        _activation = activation;
        _kernel = kernel;
        _sliceGenerator = new SliceGenerator(_kernel, depth, stride);
        _slices = new Stack<List<Slice>>();
        _inputs = new Stack<Tensor>();
        _outputs = new Stack<Tensor>();
        _filters = new Tensor[count];
        _filterGradients = new Tensor[count];
        _bias = Tensor.Random(count);
        _biasGradients = new Tensor(count);

        for (var i = 0; i < count; i++)
        {
            _filters[i] = Tensor.Random(dim, dim, depth) / (float)Math.Pow(_kernel.Dim, _kernel.Dim);
            _filterGradients[i] = Tensor.Fill(new Shape(dim, dim, depth), 0f);
        }
    }

    public override Tensor Predict(Tensor input)
    {
        var slices = _sliceGenerator.Slice(input).ToList();
        return Convolve(slices);
    }

    public override Tensor FeedForward(Tensor input)
    {
        var slices = _sliceGenerator.Slice(input).ToList();
        var result = Convolve(slices);
     
        _inputs.Push(input);
        _slices.Push(slices);
        _outputs.Push(result);
        
        return result;
    }

    public override Tensor PassBackward(Tensor errors)
    {
        var prevOut = _activation.Deactivate(_outputs.Pop());
        var previousSlices = _slices.Pop();

        Parallel.ForEach(previousSlices, slice =>
        {
            var (prevInSlice, j, k, _) = slice;
            for (var i = 0; i < _filters.Length; i++)
            {
                var (gHeight, gWidth, gDepth) = _filterGradients[i].Shape;
                for (var l = 0; l < gHeight; l++)
                {
                    for (var m = 0; m < gWidth; m++)
                    {
                        for (var t = 0; t < gDepth; t++)
                        {
                            _filterGradients[i][l, m, t] += errors[j, k, i] * prevInSlice[l, m, t] * prevOut[j, k, i];
                        }
                    }
                }
            }
        });

        return CalculateError(errors, prevOut);
    }

    public override void UpdateWeights(GradientInfo gradientInfo, int epoch, int batchSize)
    {
        var gradient = GradientFactory.Get(gradientInfo);
        for (var i = 0; i < _filters.Length; i++)
        {
            _filters[i].Add(gradient.Calculate(_filterGradients[i] / batchSize, epoch));
            _filterGradients[i].Zero();
        }
        
        var deltas = gradient.Calculate(_biasGradients / batchSize, epoch);
        _bias.Zero();
        _bias.Add(deltas);
        _biasGradients.Zero();
    }

    public override LayerWrap Save() => new()
    {
        LayerType = LayerType.Conv,
        Conv = new ConvWrap
        {
            Shape = Shape,
            Kernel = _kernel,
            Activation = _activation.GetType(),
            Stride = _sliceGenerator.Stride,
            Filters = _filters,
            FilterGradients = _filterGradients,
            Bias = _bias,
            BiasGradients = _biasGradients
        }
    };

    private Tensor Convolve(List<Slice> slices)
    {
        var output = new Tensor(Shape.Height, Shape.Width, _kernel.Count);

        Parallel.For(0, _kernel.Count, i =>
        {
            var currentKernel = _filters[i];
            var currentBias = _bias[i];
            
            foreach (var (slice, sHeight, sWidth, _) in slices)
            {
                output[sHeight, sWidth, i] += Tensor.Dot(slice, currentKernel) + currentBias;
            }
        });

        return _activation.Activate(output);
    }

    private Tensor CalculateError(Tensor errors, Tensor prevOut)
    {
        var prevInput = _inputs.Pop();
        var kernelPlanes = GetKernelPlanes();
        var output = Tensor.Like(prevInput.Shape);

        Parallel.ForEach(_sliceGenerator.Slice(errors), slice =>
        {
            var (lossSlice, j, k, _) = slice;
            for (var i = 0; i < _filters.Length; i++)
            {
                for (var l = 0; l < output.Shape.Depth; l++)
                {
                    foreach (var kPlane in kernelPlanes[i])
                    {
                        output[j, k, l] += lossSlice.PlaneDot(kPlane, l);
                    }    
                }
                
                _biasGradients[i] += errors[j, k, i] * prevOut[j, k, i];
            }
        });

        return output;
    }

    private Tensor[][] GetKernelPlanes()
    {
        var (_, _, depth) = Shape;
        var kernelPlanes = new Tensor[_kernel.Count][];

        for (var i = 0; i < _kernel.Count; i++)
        {
            var planes = new Tensor[depth];
            for (var l = 0; l < depth; l++)
            {
                planes[l] = _filters[i].Plane(l);
            }   
            kernelPlanes[i] = planes;
        }

        return kernelPlanes;
    }

}
