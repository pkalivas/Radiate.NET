using FluentAssertions;
using Radiate.NET.Domain.Activation;
using Radiate.NET.Optimizers.Perceptrons.Layers;
using Radiate.NET.UnitTests.Utils;
using Xunit;

namespace Radiate.NET.UnitTests.Layers
{
    public class ConvTests
    {
        [Fact]
        public void Conv_Output_Shape()
        {
            var layer = new Conv(LayerUtils.EightEightThreeShape, LayerUtils.FiveThreeKernel, LayerUtils.StrideOne, new ReLU());
            var output = layer.FeedForward(LayerUtils.EightEightThreeTensor);

            var (oHeight, oWidth, oDepth) = output.Shape;
            
            oDepth.Should().Be(5);
        }
    }
}
