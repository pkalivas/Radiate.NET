using System.Threading.Tasks;
using Radiate.NET.Domain.Gradients;
using Radiate.NET.Domain.Records;
using Radiate.NET.Domain.Tensors;

namespace Radiate.NET.Optimizers.Perceptrons.Layers
{
    public abstract class Layer
    {
        protected Shape Shape { get; set; }

        protected Layer(Shape shape)
        {
            Shape = shape;
        }
        
        
        public abstract Tensor Predict(Tensor pass);
        public abstract Tensor FeedForward(Tensor pass);
        public abstract Tensor PassBackward(Tensor pass);
        public abstract Task UpdateWeights(GradientInfo gradient, int epoch);
        
    }
}