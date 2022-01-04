using Radiate.Domain.Gradients;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;  

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
    public abstract void UpdateWeights(GradientInfo gradient, int epoch);
    public abstract LayerWrap Save();
}
