using Radiate.Domain.Activation;
using Radiate.Optimizers.Supervised.Perceptrons.Layers;
using Radiate.UnitTests.Utils;

namespace Radiate.UnitTests.Layers;

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

