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

        public static List<float> Product(IEnumerable<float> one, IEnumerable<float> two) =>
            one.Zip(two)
                .Select(pair => pair.First * pair.Second)
                .ToList();

        public static (float loss, List<float> errors) GetLoss(List<float> one, List<float> two, LossFunction lossFunction)
        {
            if (one.Count != two.Count)
            {
                throw new Exception("Loss vector shapes don't match");
            }
            //
            var difference = Subtract(one, two);
            return (difference.Sum(), difference);
            // var squaredError = 0f;
            // var result = one.Zip(two)
            //     .Select(pair =>
            //     {
            //         var error = (float)Math.Pow(pair.First - pair.Second, 2);
            //         squaredError += error;
            //         return error;
            //     })
            //     .ToList();
            //
            // return ((1f / one.Count()) * squaredError, result);
        }
    }
}