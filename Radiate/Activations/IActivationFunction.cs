using Radiate.Tensors;

namespace Radiate.Activations;

public interface IActivationFunction
{
    Tensor Activate(Tensor values);
    Tensor Deactivate(Tensor values);
    float Activate(float value);
    float Deactivate(float value);
    Activation ActivationType();
}