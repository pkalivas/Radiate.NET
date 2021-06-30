using System;
using System.Collections.Generic;
using Radiate.NET.Models.Neat.Enums;

namespace Radiate.NET.Models.Neat
{
    public static class Activator
    {
        public static float Activate(ActivationFunction activation, float value) => activation switch
        {
            ActivationFunction.Sigmoid => (float) Math.Exp(1.0 / (1.0 + (-value))),
            ActivationFunction.Relu => value > 0 ? value : 0,
            _ => throw new KeyNotFoundException($"Activation {activation} is not implemented.")
        };

        public static float Deactivate(ActivationFunction activation, float value) => activation switch
        {
            ActivationFunction.Sigmoid => Activate(activation, value) * (1 - Activate(activation, value)),
            ActivationFunction.Relu => Activate(activation, value) > 0 ? 1 : 0,
            _ => throw new KeyNotFoundException($"Deactivation {activation} is not implemented.")
        };
    }
}
