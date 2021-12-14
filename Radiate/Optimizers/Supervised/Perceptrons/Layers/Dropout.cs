using System;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Domain.Gradients;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons.Layers
{
    public class Dropout : Layer
    {
        private readonly float _dropoutRate;
        private readonly Random _random;

        public Dropout(float dropoutRate) : base(new Shape(0, 0, 0))
        {
            _dropoutRate = dropoutRate;
            _random = new Random();
        }

        public override Tensor Predict(Tensor input) => input;

        public override Tensor FeedForward(Tensor input) =>
            input.Read1D()
                .Select(ins => _random.NextDouble() < _dropoutRate ? 0 : ins)
                .ToArray()
                .ToTensor();

        public override Tensor PassBackward(Tensor errors) => errors;

        public override Task UpdateWeights(GradientInfo info, int epoch) => Task.CompletedTask;
    }
}