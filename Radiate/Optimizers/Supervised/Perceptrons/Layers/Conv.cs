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
    private readonly Stack<List<(Tensor slice, int sHeight, int sWidth)>> _slices;
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
        _slices = new Stack<List<(Tensor slice, int sHeight, int sWidth)>>();
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

    public override Tensor Predict(Tensor input)
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
        
        _slices.Push(slices);
        
        return _activation.Activate(output);
    }

    public override Tensor FeedForward(Tensor input)
    {
        _inputs.Push(input);

        var result = Predict(input);
        
        _outputs.Push(result);
        return result;
    }

    public override async Task<Tensor> PassBackward(Tensor errors)
    {
        var errorSliceTask = Task.Run(() => _sliceGenerator.Slice(errors));
        var prevInput = _inputs.Pop();
        var prevOut = _activation.Deactivate(_outputs.Pop());
        var previousSlices = _slices.Pop();
        
        var output = Tensor.Like(prevInput.Shape);
        foreach (var (prevInSlice, j, k) in previousSlices)
        {
            for (var i = 0; i < _filters.Length; i++)
            {
                _filterGradients[i] += errors[j, k, i] * prevInSlice * prevOut[j, k, i];
            }
        }
        
        foreach (var (lossSlice, j, k) in await errorSliceTask)
        {
            for (var i = 0; i < _filters.Length; i++)
            {
                var kernel = _filters[i];
                for (var l = 0; l < output.Shape.Depth; l++)
                {
                    output[j, k, l] += Tensor.Sum(lossSlice, kernel);
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

}
