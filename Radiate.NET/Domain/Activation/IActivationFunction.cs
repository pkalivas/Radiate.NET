using Radiate.NET.Domain.Tensors;

namespace Radiate.NET.Domain.Activation
{
    public interface IActivationFunction
    {
        Tensor Activate(Tensor values);
        Tensor Deactivate(Tensor values);
        float Activate(float value);
        float Deactivate(float value);

    }
}