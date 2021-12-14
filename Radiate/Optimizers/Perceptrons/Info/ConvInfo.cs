using Radiate.Domain.Activation;
using Radiate.Domain.Records;

namespace Radiate.Optimizers.Perceptrons.Info
{
    public class ConvInfo : LayerInfo
    {
        public Shape Shape { get; set; }
        public Kernel Kernel { get; set; }
        public Activation Activation { get; set; }
        public int Stride { get; set; } = 1;

        public ConvInfo(Shape shape, Kernel kernel, Activation activation) : this(shape, kernel, 1, activation) { }

        public ConvInfo(Shape shape, Kernel kernel, int stride, Activation activation)
        {
            Shape = shape;
            Kernel = kernel;
            Activation = activation;
            Stride = stride;
        }
    }
}