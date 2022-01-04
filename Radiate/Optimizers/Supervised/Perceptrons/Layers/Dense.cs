using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons.Layers;

public class Dense : Layer
{
    private readonly IActivationFunction _activation;
    private readonly Stack<Tensor> _inputs;
    private readonly Stack<Tensor> _outputs;
    private readonly Tensor _weights;
    private readonly Tensor _bias;
    private readonly Tensor _weightGradients;
    private readonly Tensor _biasGradients;

    public Dense(DenseWrap wrap) : base(wrap.Shape)
    {
        _activation = ActivationFunctionFactory.Get(wrap.Activation);
        _inputs = new Stack<Tensor>();
        _outputs = new Stack<Tensor>();
        _weights = wrap.Weights;
        _bias = wrap.Bias;
        _weightGradients = wrap.WeightGradients;
        _biasGradients = wrap.BiasGradients;
    }
    
    public Dense(Shape shape, IActivationFunction activation) : base(shape)
    {
        _activation = activation;
        _inputs = new Stack<Tensor>();
        _outputs = new Stack<Tensor>();
        _weights = Tensor.Random(shape.Width, shape.Height);
        _bias = Tensor.Random(shape.Width);
        _weightGradients = new Tensor(shape.Width, shape.Height);
        _biasGradients = new Tensor(shape.Width);
    }

    public override Tensor Predict(Tensor input)
    {
        if (input.Shape.Height != Shape.Height)
        {
            throw new Exception($"Input shape of {input.Shape} does not match Dense layer {Shape}");
        }
        
        if (input.Shape.Width > 0)
        {
            throw new Exception($"Cannot pass multi-dimensional tensor to dense layer.");
        }

        var result = new Tensor(Shape.Width);
        for (var i = 0; i < Shape.Width; i++)
        {
            result[i] = _bias[i] + input.Read1D()
                .Select((inVal, idx) => _weights[i, idx] * inVal)
                .Sum();
        }
        
        return _activation.Activate(result);
    }
    
    public override Tensor FeedForward(Tensor input)
    {
        var output = Predict(input);
        
        _inputs.Push(input);
        _outputs.Push(output);

        return output;
    }

    public override Tensor PassBackward(Tensor pass)
    {
        var errors = pass.Read1D();
        if (errors.Length != Shape.Width)
        {
            throw new Exception($"Error shape of {errors.Length} does not match Dense layer {Shape}.");
        }
        
        var output = _outputs.Pop();
        var input = _inputs.Pop();
        var grads = _activation.Deactivate(output);
        
        var resultError = new Tensor(Shape.Height);
        Parallel.For(0, Shape.Width, i =>
        {
            _biasGradients[i] += grads[i] * errors[i];
            
            for (var j = 0; j < Shape.Height; j++)
            {
                _weightGradients[i, j] += grads[i] * errors[i] * input[j];
                resultError[j] += _weights[i, j] * errors[i];
            }
        });

        return resultError;
    }

    public override void UpdateWeights(GradientInfo info, int epoch)
    {
        var gradient = GradientFactory.Get(info);
        
        _weights.Add(gradient.Calculate(_weightGradients, epoch));
        _bias.Add(gradient.Calculate(_biasGradients, epoch));

        _biasGradients.Zero();
        _weightGradients.Zero();
    }
    
    public override LayerWrap Save() => new()
    {
        LayerType = LayerType.Dense,
        Dense = new DenseWrap
        {
            Shape = Shape,
            Activation = _activation.GetType(),
            Weights = _weights,
            WeightGradients = _weightGradients,
            Bias = _bias,
            BiasGradients = _biasGradients
        }
    };
}
