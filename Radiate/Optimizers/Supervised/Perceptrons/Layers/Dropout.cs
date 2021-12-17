using Radiate.Domain.Gradients;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons.Layers;

public class Dropout : Layer
{
    private readonly float _dropoutRate;
    private readonly Random _random;

    public Dropout(DropoutWrap wrap) : base(new Shape(0))
    {
        _dropoutRate = wrap.DropoutRate;
        _random = new Random();
    }
    
    public Dropout(float dropoutRate) : base(new Shape(0, 0, 0))
    {
        _dropoutRate = dropoutRate;
        _random = new Random();
    }

    public override Tensor Predict(Tensor input) => input;

    public override Tensor FeedForward(Tensor input) =>
        input.Read1D()
            .Select(ins => _random.NextDouble() < _dropoutRate ? 0 : ins)
            .ToTensor();

    public override Task<Tensor> PassBackward(Tensor errors) => Task.Run(() => errors);

    public override Task UpdateWeights(GradientInfo info, int epoch) => Task.CompletedTask;

    public override LayerWrap Save() => new()
    {
        LayerType = LayerType.Dropout,
        Dropout = new DropoutWrap
        {
            DropoutRate = _dropoutRate
        }
    };
}
