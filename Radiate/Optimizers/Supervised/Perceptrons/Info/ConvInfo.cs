using Radiate.Activations;
using Radiate.Records;

namespace Radiate.Optimizers.Supervised.Perceptrons.Info;

public class ConvInfo : LayerInfo
{
    public Shape Shape { get; set; }
    public Kernel Kernel { get; set; }
    public Activation Activation { get; set; }
    public int Stride { get; set; } = 1;

    public ConvInfo(int kernelCount, int kernelDim)
    {
        Shape = new Shape(0);
        Kernel = new Kernel(kernelCount, kernelDim);
        Activation = Activation.ReLU;
    }
    
    public ConvInfo(Shape shape, Kernel kernel, Activation activation) : this(shape, kernel, 1, activation) { }

    public ConvInfo(Shape shape, Kernel kernel, int stride, Activation activation)
    {
        Shape = shape;
        Kernel = kernel;
        Activation = activation;
        Stride = stride;
    }
}
