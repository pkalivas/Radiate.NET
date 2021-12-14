using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.NET.Domain.Activation;
using Radiate.NET.Domain.Gradients;
using Radiate.NET.Domain.Records;
using Radiate.NET.Domain.Tensors;

namespace Radiate.NET.Optimizers.Perceptrons.Layers
{
    public class Dense : Layer
    {
        private readonly IActivationFunction _activation;
        private readonly Stack<Tensor> _inputs;
        private readonly Stack<Tensor> _outputs;

        private readonly Tensor _weights;
        private readonly Tensor _bias;
        private Tensor _weightGradients;
        private Tensor _biasGradients;

        
        public Dense(Shape shape, IActivationFunction activation) : base(shape)
        {
            _activation = activation;
            _weights = Tensor.Random2D(shape.Width, shape.Height);
            _bias = Tensor.Random1D(shape.Width);
            _inputs = new Stack<Tensor>();
            _outputs = new Stack<Tensor>();
            _weightGradients = new float[shape.Width, shape.Height].ToTensor();
            _biasGradients = new float[shape.Width].ToTensor();
        }
        
        public override Tensor Predict(Tensor input)
        {
            if (input.Shape.Height != Shape.Height)
            {
                throw new Exception($"Input shape of {input} does not match Dense layer {Shape}");
            }
            
            if (input.Shape.Width > 0)
            {
                throw new Exception($"Cannot pass multi-dimensional tensor to dense layer.");
            }
            
            var result = new float[Shape.Width].ToTensor();
            for (var i = 0; i < Shape.Width; i++)
            {
                result[i] = _bias[i] + input.ElementsOneD
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
            var errors = pass.ElementsOneD;
            if (errors.Length != Shape.Width)
            {
                throw new Exception($"Error shape of {errors.Length} does not match Dense layer {Shape}.");
            }
            
            var output = _outputs.Pop();
            var input = _inputs.Pop();
            var grads = _activation.Deactivate(output);
            
            var resultError = new float[Shape.Height];
            for (var i = 0; i < Shape.Width; i++)
            {
                _biasGradients[i] += grads[i] * errors[i];
                
                for (var j = 0; j < Shape.Height; j++)
                {
                    _weightGradients[i, j] += grads[i] * errors[i] * input[j];
                    resultError[j] += _weights[i, j] * errors[i];
                }
            }

            return new Tensor(resultError);
        }

        public override Task UpdateWeights(GradientInfo info, int epoch)
        {
            var gradient = GradientFactory.Get(info);
            
            _weights.Add(gradient.Calculate(_weightGradients, epoch));
            _bias.Add(gradient.Calculate(_biasGradients, epoch));

            _biasGradients.Zero();
            _weightGradients.Zero();
            
            return Task.CompletedTask;
        }
    }
}