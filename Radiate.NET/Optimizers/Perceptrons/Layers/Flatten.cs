using System.Threading.Tasks;
using Radiate.NET.Domain.Gradients;
using Radiate.NET.Domain.Records;
using Radiate.NET.Domain.Tensors;

namespace Radiate.NET.Optimizers.Perceptrons.Layers
{
    public class Flatten : Layer
    {
        private readonly Shape _previousShape;

        public Flatten(Shape shape, Shape previousShape) : base(shape)
        {
            _previousShape = previousShape;
        }

        public override Tensor Predict(Tensor input)
        {
            var (height, width, depth) = input.Shape;
            var newInput = new float[height * width * depth].ToTensor();

            var count = 0;
            for (var i = 0; i < height; i++)
            {
                for (var j = 0; j < width; j++)
                {
                    for (var k = 0; k < depth; k++)
                    {
                        newInput[count] = input[i, j, k];
                        count++;
                    }
                }
            }

            return newInput;
        }
        public override Tensor FeedForward(Tensor input)
        {
            var (height, width, depth) = input.Shape;
            var newInput = new float[height * width * depth].ToTensor();

            var count = 0;
            for (var i = 0; i < height; i++)
            {
                for (var j = 0; j < width; j++)
                {
                    for (var k = 0; k < depth; k++)
                    {
                        newInput[count] = input[i, j, k];
                        count++;
                    }
                }
            }

            return newInput;
        }

        public override Tensor PassBackward(Tensor errors)
        {
            var newOutput = Tensor.Like(_previousShape);
            var (height, width, depth) = newOutput.Shape;
            
            var count = 0;
            for (var i = 0; i < height; i++)
            {
                for (var j = 0; j < width; j++)
                {
                    for (var k = 0; k < depth; k++)
                    {
                        newOutput[i, j, k] = errors[count];
                        count++;
                    }
                }
            }

            return newOutput;
        }

        public override Task UpdateWeights(GradientInfo gradient, int epoch) => Task.CompletedTask;
    }
}