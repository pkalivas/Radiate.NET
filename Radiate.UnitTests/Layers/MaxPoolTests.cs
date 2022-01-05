using System;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Optimizers.Supervised.Perceptrons.Layers;
using Radiate.UnitTests.Utils;

namespace Radiate.UnitTests.Layers;

public class MaxPoolTests
{
    [Fact]
    public void MaxPool_Activation_Map_Retention()
    {
        var input = LayerUtils.EightEightOneTensor;
        var layer = new MaxPool(LayerUtils.EightEightOneShape, LayerUtils.TwoTwoKernel, LayerUtils.StrideTwo);
        var output = layer.FeedForward(input);

        var (iHeight, iWidth, iDepth) = input.Shape;
        var (oHeight, oWidth, oDepth) = output.Shape;

        iDepth.Should().Be(oDepth);
    }

    [Fact]
    public void MaxPool_Resize_Half()
    {
        var input = LayerUtils.EightEightOneTensor;
        var layer = new MaxPool(LayerUtils.EightEightOneShape, LayerUtils.TwoTwoKernel, LayerUtils.StrideTwo);
        var output = layer.FeedForward(input);

        var (iHeight, iWidth, iDepth) = input.Shape;
        var (oHeight, oWidth, oDepth) = output.Shape;

        var h = iHeight / 2;
        var w = iWidth / 2;

        h.Should().Be(oHeight);
        w.Should().Be(oWidth);
    }

    [Fact]
    public void MaxPool_Resize_Third()
    {
        var input = LayerUtils.NineNineOneTensor;
        var layer = new MaxPool(LayerUtils.NineNineOneShape, LayerUtils.FiveThreeKernel, LayerUtils.StrideThree);
        var output = layer.FeedForward(input);

        var (iHeight, iWidth, iDepth) = input.Shape;
        var (oHeight, oWidth, oDepth) = output.Shape;

        var h = iHeight / 3;
        var w = iWidth / 3;

        h.Should().Be(oHeight);
        w.Should().Be(oWidth);
    }

    [Fact]
    public void MaxPool_Picture_Retention()
    {
        var input = LayerUtils.EightEightOneTensor;
        var layer = new MaxPool(LayerUtils.EightEightOneShape, LayerUtils.TwoTwoKernel, LayerUtils.StrideTwo);
        var output = layer.FeedForward(input);
        
        var hDiff = new[] { 2, 4 };
        var wDiff = new[] { 2, 4 };
        var dDiff = new[] { 0, input.Shape.Depth };
        var tensorSlice = input.Slice(hDiff, wDiff, dDiff);

        tensorSlice.Max().Should().Be(output[1, 1, 0]);
    }

    [Fact]
    public async Task MaxPool_Can_FeedForward()
    {
        var input = (await Csv.LoadFromCsv("conv", "input")).Single();
        var convTrueOutput = (await Csv.LoadFromCsv("conv", "output")).Single();
        var maxPoolTrueOutput = (await Csv.LoadFromCsv("maxpool", "output")).Single();

        var convLayer = await LayerUtils.LoadConvFromFiles();
        var convOut = convLayer.FeedForward(input);
        
        foreach (var (aOut, lOut) in convTrueOutput.Flatten().Zip(convOut.Flatten()))
        {
            var roundAOut = Math.Round(aOut, 5);
            var roundLOut = Math.Round(lOut, 5);
            
            roundAOut.Should().Be(roundLOut);
        }

        var maxPoolLayer = await LayerUtils.LoadMaxPoolFromFiles();
        var maxPoolOut = maxPoolLayer.FeedForward(convOut);

        foreach (var (aOut, mOut) in maxPoolTrueOutput.Flatten().Zip(maxPoolOut.Flatten()))
        {
            var roundAOut = Math.Round(aOut, 5);
            var roundMOut = Math.Round(mOut, 5);

            roundAOut.Should().Be(roundMOut);
        }
    }
    
}