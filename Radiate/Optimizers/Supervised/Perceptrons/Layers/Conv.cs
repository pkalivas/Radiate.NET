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
    private readonly Queue<List<(Tensor slice, int sHeight, int sWidth)>> _slices;
    private readonly Stack<Tensor> _inputs;
    private readonly Stack<Tensor> _outputs;
    private readonly Tensor[] _filters;
    private readonly Tensor[] _filterGradients;
    private readonly Tensor _bias;
    private readonly Tensor _biasGradients;

    public Conv(Activation activation, Shape shape, Kernel kernel, int stride, Tensor[] filters, Tensor[] filterGradients, Tensor bias, Tensor biasGradients) : base(shape)
    {
        _activation = ActivationFunctionFactory.Get(activation);
        _kernel = kernel;
        _sliceGenerator = new SliceGenerator(_kernel, shape.Depth, stride);
        _slices = new Queue<List<(Tensor slice, int sHeight, int sWidth)>>();
        _inputs = new Stack<Tensor>();
        _outputs = new Stack<Tensor>();
        _filters = filters;
        _filterGradients = filterGradients;
        _bias = bias;
        _biasGradients = biasGradients;
    }
    
    public Conv(Shape shape, Kernel kernel, int stride, IActivationFunction activation) : base(shape)
    {
        var (count, dim) = kernel;
        var (_, _, depth) = shape;
        
        _activation = activation;
        _kernel = kernel;
        _sliceGenerator = new SliceGenerator(_kernel, depth, stride);
        _slices = new();
        _inputs = new Stack<Tensor>();
        _outputs = new Stack<Tensor>();
        _filters = new Tensor[count];
        _filterGradients = new Tensor[count];
        _bias = Tensor.Random1D(count);
        _biasGradients = new Tensor(count);

        for (var i = 0; i < count; i++)
        {
            _filters[i] = Tensor.Random3D(dim, dim, depth) / (float)Math.Pow(_kernel.Dim, _kernel.Dim);
            _filterGradients[i] = Tensor.Fill(new Shape(dim, dim, depth), 0f);
        }
    }

    public override Tensor Predict(Tensor input) => Convolve(input);

    public override Tensor FeedForward(Tensor input)
    {
        _inputs.Push(input);
        
        var result = Convolve(input);
        _outputs.Push(result);
        return result;
    }

    public override Tensor PassBackward(Tensor errors)
    {
        var prevInput = _inputs.Pop();
        var output = Tensor.Like(prevInput.Shape);
        var previousSlices = _slices.Dequeue();
        var prevOut = _activation.Deactivate(_outputs.Pop());
        
        foreach (var (prevInSlice, j, k) in previousSlices)
        {
            for (var i = 0; i < _filters.Length; i++)
            {
                _filterGradients[i] += errors[j, k, i] * prevInSlice * prevOut[j, k, i];
            }
        }

        
        foreach (var (lossSlice, j, k) in _sliceGenerator.Slice(errors))
        {
            for (var i = 0; i < _filters.Length; i++)
            {
                var kernel = _filters[i];
                for (var l = 0; l < output.Shape.Depth; l++)
                {
                    output[j, k, l] += Tensor.SumT(lossSlice, kernel);
                }
                
                _biasGradients[i] += errors[j, k, i] * prevOut[j, k, i];
            }
        }

        return output;
    }

    public override Task UpdateWeights(GradientInfo gradientInfo, int epoch)
    {
        var gradient = GradientFactory.Get(gradientInfo);
        for (var i = 0; i < _filters.Length; i++)
        {
            _filters[i].Add(gradient.Calculate(_filterGradients[i], epoch));
            _filterGradients[i].Zero();
        }
        
        
        var deltas = gradient.Calculate(_biasGradients, epoch);
        _bias.Zero();
        _bias.Add(deltas);
        _biasGradients.Zero();
        
        return Task.CompletedTask;
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
    
    private Tensor Convolve(Tensor input)
    {
        var output = new float[Shape.Height, Shape.Width, _kernel.Count].ToTensor();
        var slices = _sliceGenerator.Slice(input).ToList();

        for (var i = 0; i < _kernel.Count; i++)
        {
            var currentKernel = _filters[i];
            var currentBias = _bias[i];
            
            foreach (var (slice, sHeight, sWidth) in slices)
            {
                output[sHeight, sWidth, i] += Tensor.Sum(slice, currentKernel) + currentBias;
            }
        }
        
        _slices.Enqueue(slices);
        
        return _activation.Activate(output);
    }
    
}
