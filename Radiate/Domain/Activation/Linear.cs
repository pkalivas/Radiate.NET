using Radiate.Domain.Tensors;

namespace Radiate.Domain.Activation
{
    public class Linear : IActivationFunction
    {
        public Tensor Activate(Tensor values) => values;

        public Tensor Deactivate(Tensor values) => Tensor.Fill(values.Shape, 1f);

        public float Activate(float value) => value;

        public float Deactivate(float value) => 1f;
    }
}