using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Net.Data;
using Radiate.NET.Domain.Activation;
using Radiate.NET.Domain.Gradients;
using Radiate.NET.Domain.Loss;
using Radiate.NET.Optimizers;
using Radiate.NET.Optimizers.Perceptrons;
using Radiate.NET.Optimizers.Perceptrons.Info;

namespace Radiate.Net.Examples.Examples
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