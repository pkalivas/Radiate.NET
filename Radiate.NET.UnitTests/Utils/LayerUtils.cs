using Radiate.NET.Domain.Records;
using Radiate.NET.Domain.Tensors;

namespace Radiate.NET.UnitTests.Utils
{
    public static class LayerUtils
    {
        public static readonly Shape EightEightOneShape = new Shape(8, 8, 1);
        public static readonly Shape EightEightThreeShape = new Shape(8, 8, 3);
        public static readonly Shape NineNineOneShape = new Shape(9, 9, 1);
        public static readonly Shape NineNineThreeShape = new Shape(9, 9, 3);

        public static Kernel FiveThreeKernel = new Kernel(5, 3);
        public static Kernel FiveFiveKernel = new Kernel(5, 5);
        public static Kernel TwoTwoKernel = new Kernel(2, 2);
        
        public static Tensor EightEightOneTensor = Tensor.ARange(64).Reshape(EightEightOneShape);
        public static Tensor EightEightThreeTensor = Tensor.ARange(192).Reshape(EightEightThreeShape);
        public static Tensor NineNineOneTensor = Tensor.ARange(81).Reshape(NineNineOneShape);
        public static Tensor NineNineThreeTensor = Tensor.ARange(243).Reshape(NineNineThreeShape);

        public static int StrideOne = 1;
        public static int StrideTwo = 2;
        public static int StrideThree = 3;
    }
}