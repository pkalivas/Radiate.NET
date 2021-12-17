using Radiate.Domain.Gradients;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons.Layers;

public class Flatten : Layer
{
    private readonly Shape _previousShape;

    public Flatten(Shape shape, Shape previousShape) : base(shape)
    {
        _previousShape = previousShape;
    }

    public override Tensor Predict(Tensor input) => input.Flatten();

    public override Tensor FeedForward(Tensor input) => input.Flatten();

    public override Tensor PassBackward(Tensor errors) => errors.Reshape(_previousShape);

    public override Task UpdateWeights(GradientInfo gradient, int epoch) => Task.CompletedTask;
    
    public override LayerWrap Save() => new()
    {
        LayerType = LayerType.Flatten,
        Flatten = new FlattenWrap
        {
            Shape = Shape,
            PreviousShape = _previousShape
        }
    };
}
