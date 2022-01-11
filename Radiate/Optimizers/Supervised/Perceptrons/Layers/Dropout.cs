using Radiate.Domain.Extensions;
using Radiate.Domain.Gradients;
using Radiate.Domain.Models;
using Radiate.Domain.Models.Wraps;
using Radiate.Domain.RandomGenerator;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons.Layers;

public class Dropout : Layer
{
    private readonly float _dropoutRate;
    private readonly Random _random = RandomGenerator.Next;

    public Dropout(DropoutWrap wrap) : base(new Shape(0))
    {
        _dropoutRate = wrap.DropoutRate;
    }
    
    public Dropout(float dropoutRate) : base(new Shape(0, 0, 0))
    {
        _dropoutRate = dropoutRate;
    }

    public override Tensor Predict(Tensor input) => input;

    public override Tensor FeedForward(Tensor input) =>
        input
            .Select(ins => _random.NextDouble() < _dropoutRate ? 0 : ins)
            .ToTensor();

    public override Tensor PassBackward(Tensor errors) => errors;

    public override void UpdateWeights(GradientInfo info, int epoch, int batchSize) { }

    public override LayerWrap Save() => new()
    {
        LayerType = LayerType.Dropout,
        Dropout = new DropoutWrap
        {
            DropoutRate = _dropoutRate
        }
    };
}
