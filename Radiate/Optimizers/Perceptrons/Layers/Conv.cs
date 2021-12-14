using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Perceptrons.Layers
{
    public class Conv : Layer
    {
        private readonly IActivationFunction _activation;
        private readonly Kernel _kernel;
        private readonly SliceGenerator _sliceGenerator;
        private readonly Stack<Tensor> _outputs;
        private readonly Stack<Tensor> _inputs;
        private readonly Tensor[] _filters;
        private readonly Tensor[] _filterGradients;
        private Tensor _bias;
        private Tensor _biasGradients;
        
        public Conv(Shape shape, Kernel kernel, int stride, IActivationFunction activation) : base(shape)
        {
            var (count, dim) = kernel;
            var (_, _, depth) = shape;
            
            _activation = activation;
            _kernel = kernel;
            _sliceGenerator = new SliceGenerator(_kernel, depth, stride);
            _outputs = new();
            _inputs = new();
            _filters = new Tensor[count];
            _filterGradients = new Tensor[count];
            _bias = Tensor.Fill(new Shape(count, 0, 0), 0f);
            _biasGradients = Tensor.Fill(new Shape(count, 0, 0), 0f);

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
            
            var output = Convolve(input);
            
            _outputs.Push(output);
            
            return output;
        }

        public override Tensor PassBackward(Tensor errors)
        {
            var prevInput = _inputs.Pop();
            var prevOutput = _outputs.Pop();
            var output = Tensor.Like(prevInput.Shape);
            
            foreach (var (prevInSlice, j, k) in _sliceGenerator.Slice(prevInput))
            {
                for (var i = 0; i < _filters.Length; i++)
                {
                    _filterGradients[i] += errors[j, k, i] * prevInSlice;// * prevOutput[j, k, i];
                }
            }

            foreach (var (lossSlice, j, k) in _sliceGenerator.Slice(errors))
            {
                for (var i = 0; i < _filters.Length; i++)
                {
                    var kernel = _filters[i];
                    for (var l = 0; l < output.Shape.Depth; l++)
                    {
                        output[j, k, l] += Tensor.Sum(lossSlice, kernel);// *  prevOutput[j, k, l];
                    }

                    _biasGradients[i] += errors[j, k, i];
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
            
            _bias.Zero();
            _bias.Add(_biasGradients);

            _biasGradients.Zero();
            
            return Task.CompletedTask;
        }
        
        private Tensor Convolve(Tensor input)
        {
            var (hStride, wStride) = _sliceGenerator.CalcStride(input);
            var output = new float[hStride, wStride, _kernel.Count].ToTensor();

            foreach (var (slice, sHeight, sWidth) in _sliceGenerator.Slice(input))
            {
                for (var i = 0; i < _kernel.Count; i++)
                {
                    var currentKernel = _filters[i];
                    var currentBias = _bias[i];

                    output[sHeight, sWidth, i] += Tensor.Sum(slice, currentKernel) + currentBias;
                }
                
            }

            // return output;
            return _activation.Activate(output);
        }
        
    }
} 