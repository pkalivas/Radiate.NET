using System;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Activations;
using Radiate.Losses;
using Radiate.Optimizers.Supervised.Perceptrons.Layers;
using Radiate.Tensors;
using Radiate.UnitTests.Utils;

namespace Radiate.UnitTests.MultiLayerPerceptron;

public class ConvNetTests
{
    [Fact]
    public async Task ConvNet_Can_FeedForward()
    {
        var input = (await Csv.LoadFromCsv("conv", "input")).Single();
        var convTrueOutput = (await Csv.LoadFromCsv("conv", "output")).Single();
        var maxPoolTrueOutput = (await Csv.LoadFromCsv("maxpool", "output")).Single();
        var denseSoftMaxTrueIn = (await Csv.LoadFromCsv("dense", "input")).Single(); 
        var denseSoftMaxTrueOut = (await Csv.LoadFromCsv("dense", "output")).Single();

        var convLayer = await LayerUtils.LoadConvFromFiles();
        var convOut = convLayer.FeedForward(input);
        
        var maxPoolLayer = await LayerUtils.LoadMaxPoolFromFiles();
        var maxPoolOut = maxPoolLayer.FeedForward(convOut);

        var flattenLayer = new Flatten(maxPoolOut.Shape);
        var flattenOut = flattenLayer.FeedForward(maxPoolOut);

        var denseLayer = await LayerUtils.LoadDenseFromFiles(Activation.SoftMax);
        var denseOut = denseLayer.FeedForward(flattenOut);
        
        foreach (var (aOut, lOut) in convTrueOutput.Flatten().ToArray().Zip(convOut.Flatten().ToArray()))
        {
            var roundAOut = Math.Round(aOut, 5);
            var roundLOut = Math.Round(lOut, 5);
            
            roundAOut.Should().Be(roundLOut);
        }
        
        foreach (var (aOut, mOut) in maxPoolTrueOutput.Flatten().ToArray().Zip(maxPoolOut.Flatten().ToArray()))
        {
            var roundAOut = Math.Round(aOut, 5);
            var roundMOut = Math.Round(mOut, 5);

            roundAOut.Should().Be(roundMOut);
        }
        
        foreach (var (aOut, lOut) in denseSoftMaxTrueIn.Flatten().ToArray().Zip(flattenOut.ToArray()))
        {
            var roundAOut = Math.Round(aOut, 5);
            var roundLOut = Math.Round(lOut, 5);
            
            roundAOut.Should().Be(roundLOut);
        }
        
        foreach (var (aOut, lOut) in denseSoftMaxTrueOut.Flatten().ToArray().Zip(denseOut.ToArray()))
        {
            var roundAOut = Math.Round(aOut, 5);
            var roundLOut = Math.Round(lOut, 5);
            
            roundAOut.Should().Be(roundLOut);
        }
    }

    [Fact]
    public async Task ConvNet_Can_Backprop()
    {
        var input = (await Csv.LoadFromCsv("conv", "input")).Single();

        var convLayer = await LayerUtils.LoadConvFromFiles();
        var convOut = convLayer.FeedForward(input);
        
        var maxPoolLayer = await LayerUtils.LoadMaxPoolFromFiles();
        var maxPoolOut = maxPoolLayer.FeedForward(convOut);

        var flattenLayer = new Flatten(maxPoolOut.Shape);
        var flattenOut = flattenLayer.FeedForward(maxPoolOut);

        var denseLayer = await LayerUtils.LoadDenseFromFiles(Activation.SoftMax);
        var denseOut = denseLayer.FeedForward(flattenOut);

        var lossFunc = new Difference();
        var target = Tensor.Fill(denseOut.Shape, 0f);
        target[3] = 1;
        
        var errors = lossFunc.Calculate(denseOut, target).Errors;

        var denseError = denseLayer.PassBackward(errors);
        var flattenError = flattenLayer.PassBackward(denseError);
        var maxPoolError = maxPoolLayer.PassBackward(flattenError);
        var convError = convLayer.PassBackward(maxPoolError);

        var hasErrors = convError.Max();
        hasErrors.Should().NotBe(null);
    }
}