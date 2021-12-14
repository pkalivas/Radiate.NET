using Radiate.Domain.Tensors;

namespace Radiate.Domain.Gradients
{
    public interface IGradient
    {
        public Tensor Calculate(Tensor gradients, int epoch);
    }
}