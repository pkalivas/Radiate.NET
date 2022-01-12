using Radiate.Gradients;
using Radiate.IO.Wraps;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons.Layers;

public abstract class Layer
{
    public Shape Shape { get; set; }

    protected Layer(Shape shape)
    {
        Shape = shape;
    }
    
    
    public abstract Tensor Predict(Tensor pass);
    public abstract Tensor FeedForward(Tensor pass);
    public abstract Tensor PassBackward(Tensor pass);
    public abstract void UpdateWeights(GradientInfo gradient, int epoch, int batchSize);
    public abstract LayerWrap Save();
}
