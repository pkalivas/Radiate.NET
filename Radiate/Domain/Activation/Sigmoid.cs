﻿using System;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Activation
{
    public class Sigmoid : IActivationFunction
    {
        private const float MinClipValue = -5f;
        private const float MaxClipValue = 5f;
        
        public Tensor Activate(Tensor values) => Tensor.Apply(values, Calc); 

        public Tensor Deactivate(Tensor values) => Tensor.Apply(values, val =>  val * (1 - val) switch
        {
            > MaxClipValue => MaxClipValue,
            < MinClipValue => MinClipValue,
            var x and >= MinClipValue and <= MaxClipValue => x,
            var x => throw new Exception($"Failed to activate Sigmoid {x}")
        });

        public float Activate(float value) => Calc(value);

        public float Deactivate(float value) => DCalc(value);


        private static float Calc(float val) => 1f / (1f + (float)Math.Exp(-val));

        private static float DCalc(float val) => val * (1 - val) switch
        {
            > MaxClipValue => MaxClipValue,
            < MinClipValue => MinClipValue,
            var x and >= MinClipValue and <= MaxClipValue => x,
            var x => throw new Exception($"Failed to activate Sigmoid {x}")
        };
    }
}