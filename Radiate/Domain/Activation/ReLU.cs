using Radiate.Domain.Tensors;

namespace Radiate.Domain.Activation;

public class ReLU : IActivationFunction
{
    public Tensor Activate(Tensor values) => Tensor.Apply(values, val => val > 0 ? val : 0);

    public Tensor Deactivate(Tensor values) => Tensor.Apply(values, val => val > 0 ? 1f : 0f);
    
    public float Activate(float value) => value > 0 ? value : 0;

    public float Deactivate(float value) => value > 0 ? 1f : 0f;
}