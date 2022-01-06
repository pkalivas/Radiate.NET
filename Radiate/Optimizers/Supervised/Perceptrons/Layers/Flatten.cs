using Radiate.Domain.Gradients;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons.Layers;

public class Flatten : Layer
{
    public Flatten(FlattenWrap wrap) : base(wrap.Shape) { }
    
    public Flatten(Shape shape) : base(shape) { }

    public override Tensor Predict(Tensor input) => input.Flatten();

    public override Tensor FeedForward(Tensor input) => input.Flatten();

    public override Tensor PassBackward(Tensor errors) => errors.Reshape(Shape);

    public override void UpdateWeights(GradientInfo gradient, int epoch, int batchSize) { }
    
    public override LayerWrap Save() => new()
    {
        LayerType = LayerType.Flatten,
        Flatten = new FlattenWrap { Shape = Shape }
    };
}
