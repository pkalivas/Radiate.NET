using System;
using System.Collections.Generic;
using Radiate.NET.Models.Neat.Enums;

namespace Radiate.NET.Models.Neat
{
    public static class Activator
    {
        public static float Activate(ActivationFunction activation, float value) => activation switch
        {
            ActivationFunction.Sigmoid => 1f / (1f + (float) Math.Exp(-value * 4.9)),
            ActivationFunction.Relu => value > 0 ? value : 0,
            ActivationFunction.Tanh => (float) Math.Tanh(value),
            _ => throw new KeyNotFoundException($"Activation {activation} is not implemented.")
        };

        public static float Deactivate(ActivationFunction activation, float value) => activation switch
        {
            ActivationFunction.Sigmoid => Activate(activation, value) * (1 - Activate(activation, value)),
            ActivationFunction.Relu => Activate(activation, value) > 0 ? 1 : 0,
            ActivationFunction.Tanh => 1f - (float) Math.Pow(Activate(activation, value), 2),
            _ => throw new KeyNotFoundException($"Deactivation {activation} is not implemented.")
        };
    }
}
