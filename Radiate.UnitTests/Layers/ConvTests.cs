using System;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Activations;
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

        var (_, _, oDepth) = output.Shape;

        oDepth.Should().Be(5);
    }

    [Fact]
    public async Task Conv_Can_FeedForward()
    {
        var input = (await Csv.LoadFromCsv("conv", "input")).Single();
        var output = (await Csv.LoadFromCsv("conv", "output")).Single();

        var layer = await LayerUtils.LoadConvFromFiles();
        var layerOut = layer.FeedForward(input);
        
        foreach (var (aOut, lOut) in output.Flatten().Zip(layerOut.Flatten()))
        {
            var roundAOut = Math.Round(aOut, 5);
            var roundLOut = Math.Round(lOut, 5);
            
            roundAOut.Should().Be(roundLOut);
        }
    }

}