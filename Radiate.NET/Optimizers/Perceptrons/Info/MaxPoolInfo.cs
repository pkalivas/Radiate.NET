using Radiate.NET.Domain.Records;

namespace Radiate.NET.Optimizers.Perceptrons.Info
{
    public class MaxPoolInfo : LayerInfo
    {
        public Kernel Kernel { get; set; }
        public int Stride { get; set; }

        public MaxPoolInfo(Kernel kernel, int stride = 1)
        {
            Kernel = kernel;
            Stride = stride;
        }
    }
}