using System;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Net.Data;
using Radiate.NET.Models.Neat;
using Radiate.NET.Models.Neat.Enums;
using Radiate.NET.Models.Neat.Layers;

namespace Radiate.Net.Examples.Examples
{
    public class TrainDense : IExample
    {
        public Task Run()
        {
            var (inputs, target) = new XOR().GetDataSet();
            
            var neat = new Neat()
                .AddLayer(new Dense(2, 16, ActivationFunction.Relu))
                .AddLayer(new Dense(16, 1, ActivationFunction.Sigmoid));

            neat.Train(inputs, target, .1f, (epoch, loss) => epoch == 200);

            foreach (var (point, idx) in inputs.Select((val, idx) => (val, idx)))
            {
                var output = neat.Forward(point);
                Console.WriteLine($"Input ({point[0]} {point[1]}) output ({output[0]} answer ({target[idx][0]})");
            }

            return Task.CompletedTask;
        }
    }
}