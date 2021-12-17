using Radiate.Domain.Records;

namespace Radiate.Optimizers.Supervised.Perceptrons.Info;

public class MaxPoolInfo : LayerInfo
{
    public int Stride { get; set; }

    public Kernel Kernel;

    public MaxPoolInfo(int kernelCount, int kernelDim, int stride = 1)
    {
        Kernel = new Kernel(kernelCount, kernelDim);
        Stride = stride;
    }
}
