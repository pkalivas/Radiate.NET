using System.Collections.Generic;
using System.Threading.Tasks;
using Radiate.Domain.Gradients;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons.Layers
{
    public class MaxPool : Layer
    {
        private readonly Kernel _kernel;
        private readonly Stack<Tensor> _inputs;
        private readonly Tensor _gradients;
        private readonly SliceGenerator _sliceGenerator;
        private readonly int _stride;
        
        public MaxPool(Shape shape, Kernel kernel, int stride) : base(shape)
        {
            _kernel = kernel;
            _inputs = new();
            _gradients = Tensor.Fill(shape, 0f);
            _stride = stride;
            _sliceGenerator = new SliceGenerator(kernel, shape.Depth, stride);
        }

        public override Tensor Predict(Tensor input) => Pool(input);

        public override Tensor FeedForward(Tensor  input)
        {
            var pooledResult = Pool(input);
            
            _inputs.Push(input);

            return pooledResult;
        }

        public override Tensor PassBackward(Tensor  errors)
        {
            var prevInput = _inputs.Pop();
            var output = Tensor.Fill(prevInput.Shape, 0f);

            foreach (var (slice, sHeight, sWidth, sDepth) in _sliceGenerator.Slice3D(prevInput))
            {
                var (height, width, depth) = slice.Shape;
                var sliceMax = slice.Max();
                
                for (var i = 0; i < height; i++)
                {
                    for (var j = 0; j < width; j++)
                    {
                        for (var k = 0; k < depth; k++)
                        {
                            if (slice[i, j, k] == sliceMax)
                            {
                                var deltaHeight = sHeight * _stride + i;
                                var deltaWidth = sWidth * _stride + j;

                                output[deltaHeight, deltaWidth, k] = errors[i, j, sDepth];
                            }
                        }
                    }
                }
            }
            
            return output;
        }

        public override Task UpdateWeights(GradientInfo gradient, int epoch)
        {
            return Task.CompletedTask;
        }

        private Tensor Pool(Tensor input)
        {
            var (hStride, wStride) = _sliceGenerator.CalcStride(input);
            var output = new float[hStride, wStride, _gradients.Shape.Depth].ToTensor();

            foreach (var (slice, sHeight, sWidth, sDepth) in _sliceGenerator.Slice3D(input))
            {
                output[sHeight, sWidth, sDepth] = slice.Max();
            }
            
            return output;
        }
        
    }
}