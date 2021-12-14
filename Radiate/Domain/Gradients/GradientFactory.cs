using System;

namespace Radiate.Domain.Gradients
{
    public static class GradientFactory
    {
        public static IGradient Get(GradientInfo info) => info.Gradient switch
        {
            Gradient.SGD => new SGD(info.LearningRate),
            Gradient.Adam => new Adam(info.LearningRate, info.BetaOne, info.BetaTwo, info.Epsilon),
            _ => throw new Exception($"{info.Gradient} is not implemented.")
        };
    }
}