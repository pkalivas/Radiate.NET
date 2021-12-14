using System;
using System.Linq;
using Radiate.NET.Domain.Tensors;

namespace Radiate.NET.Domain.Activation
{
    public class SoftMax : IActivationFunction
    {
        private const float MaxClipValue = 200;
        private const float MinClipValue = -200;
        
        public Tensor Activate(Tensor values)
        {
            var expSum = values.ElementsOneD.Sum(val => (float)Math.Exp(val));
            
            return values.ElementsOneD
                .Select(val => (float)Math.Exp(val) / expSum)
                .ToArray()
                .ToTensor();
        }

        public Tensor Deactivate(Tensor values)
        {
            var diagMatrix = values.ElementsOneD
                .Select((val, idx) => Enumerable
                    .Range(0, values.ElementsOneD.Length)
                    .Select(num => num == idx ? val : 0)
                    .ToArray())
                .ToArray();
            
            var tiledMatrix = values.ElementsOneD
                .Select(val => Enumerable
                    .Range(0, values.ElementsOneD.Length)
                    .Select(_ => val)
                    .ToArray())
                .ToArray();
            
            var transposedMatrix = values.ElementsOneD
                .Select(_ => values.ElementsOneD.Select(val => val).ToArray())
                .ToArray();
            
            var result = new float[values.ElementsOneD.Length];
            for (var i = 0; i < values.ElementsOneD.Length; i++)
            {
                for (var j = 0; j < values.ElementsOneD.Length; j++)
                {
                    result[i] += diagMatrix[j][i] - (tiledMatrix[j][i] * transposedMatrix[j][i]);
                }
            }
            
            
            return result.Select(val => val switch
            {
                > MaxClipValue => MaxClipValue,
                < MinClipValue => MinClipValue,
                var x and >= MinClipValue and <= MaxClipValue => x,
                var x => throw new Exception($"Failed to activate Softmax {x}")
            }).ToArray().ToTensor();
        }

        public float Activate(float value) => throw new Exception($"Softmax of single value is not real.");

        public float Deactivate(float value) => throw new Exception($"Cannot take dSoftmax of single value");
    }
}