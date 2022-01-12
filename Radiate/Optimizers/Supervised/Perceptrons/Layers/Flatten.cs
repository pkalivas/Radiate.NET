using Radiate.Gradients;
using Radiate.IO.Wraps;
using Radiate.Records;
using Radiate.Tensors;

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
