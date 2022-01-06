using Radiate.Domain.Records;

namespace Radiate.Optimizers.Supervised.Perceptrons.Info;

public class MaxPoolInfo : LayerInfo
{
    public int Stride { get; set; }

    public Kernel Kernel;

    public MaxPoolInfo(int stride = 1, int kernelCount = 5, int kernelDim = 3)
    {
        Kernel = new Kernel(kernelCount, kernelDim);
        Stride = stride;
    }
}
