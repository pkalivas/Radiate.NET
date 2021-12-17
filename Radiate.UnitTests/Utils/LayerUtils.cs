using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Domain.Activation;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Supervised.Perceptrons.Layers;

namespace Radiate.UnitTests.Utils;

public static class LayerUtils
{
    public static readonly Shape EightEightOneShape = new(8, 8, 1);
    public static readonly Shape EightEightThreeShape = new(8, 8, 3);
    public static readonly Shape NineNineOneShape = new(9, 9, 1);
    public static readonly Shape NineNineThreeShape = new(9, 9, 3);

    public static readonly Kernel FiveThreeKernel = new(5, 3);
    public static readonly Kernel FiveFiveKernel = new(5, 5);
    public static readonly Kernel TwoTwoKernel = new(2, 2);
    
    public static readonly Tensor EightEightOneTensor = Tensor.ARange(64).Reshape(EightEightOneShape);
    public static readonly Tensor EightEightThreeTensor = Tensor.ARange(192).Reshape(EightEightThreeShape);
    public static readonly Tensor NineNineOneTensor = Tensor.ARange(81).Reshape(NineNineOneShape);
    public static readonly Tensor NineNineThreeTensor = Tensor.ARange(243).Reshape(NineNineThreeShape);

    public const int StrideOne = 1;
    public const int StrideTwo = 2;
    public const int StrideThree = 3;

    public static async Task<Layer> LoadConvFromFiles()
    {
        var kernels = (await Csv.LoadFromCsv("conv", "kernel")).ToArray();
        var bias = (await Csv.LoadFromCsv("conv", "biases")).Single();
        var input = (await Csv.LoadFromCsv("conv", "input")).Single();

        return new Conv(new ConvWrap
        {
            Shape = input.Shape,
            Stride = 1,
            Kernel = new Kernel(16, 3),
            Activation = Activation.Linear,
            Bias = bias,
            BiasGradients = bias,
            Filters = kernels,
            FilterGradients = kernels.Select(kern => Tensor.Like(kern.Shape)).ToArray()
        });
    }

    public static async Task<Layer> LoadMaxPoolFromFiles() => await Task.Run(() => new MaxPool(new MaxPoolWrap
    {
        Shape = new Shape(28, 28, 16),
        Kernel = new Kernel(16, 2),
        Stride = StrideTwo
    }));

    public static async Task<Layer> LoadDenseFromFiles(Activation activation)
    {
        var weights = (await Csv.LoadFromCsv("dense", "weights")).Single();
        var biases = (await Csv.LoadFromCsv("dense", "biases")).Single();

        return new Dense(new DenseWrap
        {
            Activation = activation,
            Bias = biases,
            Weights = weights,
            WeightGradients = Tensor.Like(weights.Shape),
            BiasGradients = Tensor.Like(biases.Shape),
            Shape = new Shape(weights.Shape.Width, weights.Shape.Height)
        });
    }


}