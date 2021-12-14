using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Radiate.NET.Domain.Gradients;
using Radiate.NET.Domain.Loss;
using Radiate.NET.Domain.Models;
using Radiate.NET.Domain.Records;
using Radiate.NET.Domain.Tensors;

namespace Radiate.NET.Optimizers.Perceptrons
{
    public interface IOptimizer
    {
        Tensor Predict(Tensor  inputs);
        Tensor  PassForward(Tensor  inputs);
        void PassBackward(Tensor  errors, int epoch);
        Task Update(GradientInfo gradient, int epoch);
    }
}