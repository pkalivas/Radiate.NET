﻿using System;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Net.Data;
using Radiate.Net.Data.Utils;
using Radiate.NET.Enums;
using Radiate.NET.Models.Neat;
using Radiate.NET.Models.Neat.Enums;
using Radiate.NET.Models.Neat.Layers;

namespace Radiate.Net.Examples.Examples
{
    public class TrainLSTM : IExample
    {
        public Task Run()
        {
            var (inputs, target) = new SimpleMemory().GetDataSet();
            
            var neat = new Neat()
                .SetBatchSize(10)
                .SetLossFunction(LossFunction.Difference)
                .AddLayer(new LSTM(1, 1, 32, ActivationFunction.Sigmoid));

            neat.Train(inputs, target, .001f, (epoch, loss) =>
            {
                Console.WriteLine($"{epoch} - {loss}");
                return epoch == 100;
            });
            
            neat.ResetGenome();

            foreach (var (point, idx) in inputs.Select((val, idx) => (val, idx)))
            {
                var output = neat.Forward(point);
                Console.WriteLine($"Input {point[0]} Expecting {target[idx][0]} Guess {output[0]}");
            }
            
            Console.WriteLine("\nTesting Memory...");
            Console.WriteLine($"Input {1f} Expecting {0f} Guess {neat.Forward(new(){ 1f })[0]}");
            Console.WriteLine($"Input {0f} Expecting {0f} Guess {neat.Forward(new(){ 0f })[0]}");
            Console.WriteLine($"Input {0f} Expecting {0f} Guess {neat.Forward(new(){ 0f })[0]}");
            Console.WriteLine($"Input {0f} Expecting {1f} Guess {neat.Forward(new(){ 0f })[0]}");
            
            return Task.CompletedTask;
        }
    }
}