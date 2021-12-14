using System;
using Radiate.NET.Domain.Tensors;

namespace Radiate.NET.Domain.Activation
{
    public class Tanh : IActivationFunction
    {
        public Tensor Activate(Tensor values) => Tensor.Apply(values, val => (float)Math.Tanh(val));

        public Tensor Deactivate(Tensor values) => Tensor.Apply(values, val => 1f - (float)Math.Pow(val, 2));

        public float Activate(float value) => (float)Math.Tanh(value);

        public float Deactivate(float value) => 1f - (float)Math.Pow(value, 2);
    }
}