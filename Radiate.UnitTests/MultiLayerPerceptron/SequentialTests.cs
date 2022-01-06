using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Radiate.Domain.Activation;
using Radiate.Domain.Models;
using Radiate.Optimizers.Supervised.Perceptrons.Info;
using Radiate.UnitTests.Utils;

namespace Radiate.UnitTests.MultiLayerPerceptron;

public class SequentialTests
{
    [Fact]
    public async Task MultiLayerPerceptron_Can_Save()
    {
        var outputSize = 10;
        var neuralNetwork = new Optimizers.Supervised.Perceptrons.MultiLayerPerceptron()
            .AddLayer(new ConvInfo(16, 3))
            .AddLayer(new MaxPoolInfo(16, 3) { Stride = 2})
            .AddLayer(new FlattenInfo())
            .AddLayer(new DenseInfo(64, Activation.Sigmoid))
            .AddLayer(new DenseInfo(outputSize, Activation.SoftMax));

        neuralNetwork.PassForward(LayerUtils.NineNineThreeTensor, LayerUtils.NineNineThreeTensor);
        
        var baseDir = $"{Environment.CurrentDirectory}\\Data\\saves";
        var savedFileName = $"{baseDir}\\convnet.json";
        
        var wrapped = neuralNetwork.Save();
        var serialized = JsonConvert.SerializeObject(wrapped);
        await File.WriteAllTextAsync(savedFileName, serialized);

        var fileNames = Directory.GetFiles(baseDir)
            .Where(file => file == savedFileName)
            .ToList();

        fileNames.Should().Contain(savedFileName);
    }

    [Fact]
    public async Task MultiLayerPerceptron_Can_Load()
    {
        var filePath = $"{Environment.CurrentDirectory}\\Data\\saves\\convnet.json";
        var contents = await File.ReadAllTextAsync(filePath);
        
        var wrap = JsonConvert.DeserializeObject<SupervisedWrap>(contents);
        var network = new Optimizers.Supervised.Perceptrons.MultiLayerPerceptron(wrap);

        var forwardPass = network.PassForward(LayerUtils.NineNineThreeTensor, LayerUtils.NineNineThreeTensor);

        forwardPass.Result.Shape.Should().NotBe(null);
    }
}