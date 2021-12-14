using Radiate.NET.Domain.Tensors;

namespace Radiate.NET.Domain.Gradients
{
    public interface IGradient
    {
        public Tensor Calculate(Tensor gradients, int epoch);
    }
}