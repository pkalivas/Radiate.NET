using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

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
}