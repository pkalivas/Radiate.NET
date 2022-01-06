using Radiate.Data;
using Radiate.Data.Utils;
using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Domain.Tensors;
using Radiate.Optimizers;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;

namespace Radiate.Examples.Examples;

public class TrainDense : IExample
{
    public async Task Run()
    {
        const int maxEpoch = 1000;
        var progress = new ProgressBar(maxEpoch);
        
        var (inputs, targets) = await new XOR().GetDataSet();

        var pair = new TensorTrainSet(inputs, targets).Batch(1);
        
        var mlp = new MultiLayerPerceptron(new GradientInfo { Gradient = Gradient.SGD })
            .AddLayer(new DenseInfo(32, Activation.ReLU))
            .AddLayer(new DenseInfo(1, Activation.Sigmoid));

        var optimizer = new Optimizer<MultiLayerPerceptron>(mlp, pair, Loss.MSE);
        var model = await optimizer.Train(epoch =>
        {
            progress.Tick(epoch);
            return epoch.Index == maxEpoch;
        });
        
        foreach (var (ins, outs) in pair.TrainingInputs)
        {
            var pred = model.Predict(ins.First());
            Console.WriteLine($"Answer {outs[0][0]} Confidence {pred.Confidence}");
        }
    }
}