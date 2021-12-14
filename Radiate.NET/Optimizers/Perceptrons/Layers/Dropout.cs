using System;
using System.Linq;
using System.Threading.Tasks;
using Radiate.NET.Domain.Gradients;
using Radiate.NET.Domain.Records;
using Radiate.NET.Domain.Tensors;

namespace Radiate.NET.Optimizers.Perceptrons.Layers
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
            input.ElementsOneD
                .Select(ins => _random.NextDouble() < _dropoutRate ? 0 : ins)
                .ToArray()
                .ToTensor();

        public override Tensor PassBackward(Tensor errors) => errors;

        public override Task UpdateWeights(GradientInfo info, int epoch) => Task.CompletedTask;
    }
}