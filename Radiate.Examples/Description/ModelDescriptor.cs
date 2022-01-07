﻿using Radiate.Optimizers.Supervised.Perceptrons;

namespace Radiate.Examples.Description;

public static class ModelDescriptor
{
    public static string Describe<T>(T data)
    {
        if (data is MultiLayerPerceptron perceptron)
        {
            return new PerceptronDescriptor().Describe(perceptron);
        }

        return "";
    }
}