using Radiate.Tensors;

namespace Radiate.Gradients;

public interface IGradient
{
    public Tensor Calculate(Tensor gradients, int epoch);
}