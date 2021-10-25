using System;
using System.Collections.Generic;
using Radiate.NET.Models.Neat.Enums;

namespace Radiate.NET.Models.Neat
{
    public static class Activator
    {
        private const float MinClipValue = -5f;
        private const float MaxClipValue = 5f;
            
        public static float Activate(ActivationFunction activation, float value) => activation switch
        {
            ActivationFunction.Sigmoid => Sigmoid(value),
            ActivationFunction.ReLU => Relu(value),
            ActivationFunction.Tanh => Tanh(value),
            ActivationFunction.ReLU6 => Relu(value) switch
            {
                > 6f => 6f,
                var x => x
            },
            _ => throw new KeyNotFoundException($"Activation {activation} is not implemented.")
        };

        public static float Deactivate(ActivationFunction activation, float value) => activation switch
        {
            ActivationFunction.Sigmoid => DSigmoid(value) switch
            {
                > MaxClipValue => MaxClipValue,
                < MinClipValue => MinClipValue,
                var x and >= MinClipValue and <= MaxClipValue => x,
                _ => throw new Exception($"Failed to activate Sigmoid {value}")
            },
            ActivationFunction.ReLU or ActivationFunction.ReLU6=> DRelu(value),
            ActivationFunction.Tanh => DTanh(value) switch
            {   
                > MaxClipValue => MaxClipValue,
                < MinClipValue => MinClipValue,
                var x and >= MinClipValue and <= MaxClipValue => x,
                _ => throw new Exception($"Failed to deactivate Tanh {value}")
            },
            _ => throw new KeyNotFoundException($"Deactivation {activation} is not implemented.")
        };

        // Sigmoid
        private static float Sigmoid(float val) => 1f / (1f + (float)Math.Exp(-val * 4.9));
        private static float DSigmoid(float val) => Sigmoid(val) * (1 - Sigmoid(val));

        // Relu
        private static float Relu(float val) => val > 0 ? val : 0;
        private static float DRelu(float val) => Relu(val) > 0 ? 1 : 0;

        // Tanh
        private static float Tanh(float val) => (float)Math.Tanh(val);
        private static float DTanh(float val) => 1f - (float)Math.Pow(Tanh(val), 2);
    }
}
