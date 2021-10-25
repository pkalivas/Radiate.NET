using System;
using System.Collections.Generic;
using System.Linq;
using Radiate.NET.Enums;
using Radiate.NET.Models.Neat.Enums;

namespace Radiate.NET.Models.Neat
{
    public static class VectorOperations
    {
        public static List<float> ElementMultiply(IEnumerable<float> one, IEnumerable<float> two) => 
            one.Zip(two)
                .Select(pair => pair.First * pair.Second)
                .ToList();

        public static List<float> ElementAdd(IEnumerable<float> one, IEnumerable<float> two) => 
            one.Zip(two)
                .Select(pair => pair.First + pair.Second)
                .ToList();
        
        public static List<float> Subtract(IEnumerable<float> one, IEnumerable<float> two) => 
            one.Zip(two)
                .Select(pair => pair.First - pair.Second)
                .ToList();

        public static List<float> ElementActivate(IEnumerable<float> one, ActivationFunction activationFunction) => 
            one
                .Select(val => Activator.Activate(activationFunction, val))
                .ToList();

        public static List<float> ElementDeactivate(IEnumerable<float> one, ActivationFunction activationFunction) =>
            one
                .Select(val => Activator.Deactivate(activationFunction, val))
                .ToList();
        
        public static List<float> OuterProduct(IEnumerable<float> one, IEnumerable<float> two) =>
            one
                .SelectMany(val => two
                    .Select(other => val * other))
                .ToList();

        public static List<float> Invert(IEnumerable<float> one) =>
            one
                .Select(val => 1 - val)
                .ToList();

        public static List<float> Pow(IEnumerable<float> one) =>
            one
                .Select(val => (float)Math.Pow(val, 2))
                .ToList();
        
        public static (float loss, List<float> errors) GetLoss(List<float> one, List<float> two, LossFunction lossFunction)
        {
            if (one.Count != two.Count)
            {
                throw new Exception("Loss vector shapes don't match");
            }

            if (lossFunction is LossFunction.Difference)
            {
                var result = Subtract(one, two);
                return (result.Sum(), result);
            }

            if (lossFunction is LossFunction.MeanSquaredError)
            {
                var result = one.Zip(two)
                    .Select(pair => (float)Math.Pow(pair.Second - pair.First, 2))
                    .ToList();
                return (1f - (result.Sum() / one.Count), result);
            }

            throw new Exception($"Loss function {lossFunction} is not implemented");
        }
    }
}