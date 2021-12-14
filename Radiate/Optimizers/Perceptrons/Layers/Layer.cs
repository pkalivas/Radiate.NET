using System.Threading.Tasks;
using Radiate.Domain.Gradients;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Perceptrons.Layers
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