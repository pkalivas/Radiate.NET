using System;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Data;
using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Loss;
using Radiate.Optimizers;
using Radiate.Optimizers.Perceptrons;
using Radiate.Optimizers.Perceptrons.Info;

namespace Radiate.Examples.Examples
{
    public class TrainDense : IExample
    {
        public async Task Run()
        {
            var (inputs, targets) = new XOR().GetDataSet();

            var gradient = new GradientInfo { Gradient = Gradient.SGD };
            
            var mlp = new MultiLayerPerceptron(2, 1)
                .AddLayer(new DenseInfo(32, Activation.ReLU))
                .AddLayer(new DenseInfo(32, Activation.Sigmoid));
            
            var classifier = new Optimizer(mlp, Loss.MSE, gradient);
            
            await classifier.Train(inputs, targets, (epoch) => epoch.Count == 500);
            
            foreach (var (ins, outs) in inputs.Zip(targets))
            {
                var pred = classifier.Predict(ins);
                Console.WriteLine($"Answer {outs[0]} Confidence {pred.Confidence}");
            }
        }
    }
}