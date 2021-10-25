using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Net.Data;
using Radiate.Net.Data.Utils;
using Radiate.NET.Engine;
using Radiate.NET.Engine.Enums;
using Radiate.NET.Engine.ParentalCriteria;
using Radiate.NET.Engine.SurvivorCriteria;
using Radiate.NET.Enums;
using Radiate.NET.Models.Neat;
using Radiate.NET.Models.Neat.Enums;
using Radiate.NET.Models.Neat.Layers;

namespace Radiate.Net.Examples.Examples
{
    public class TemperatureTimeSeries : IExample
    {
        public async Task Run()
        {
            var dataLayerSize = 1;
            var maxEpoch = 1000;
            var evolveEpochs = 50;
            var learningRate = .001f;
            
            var (ins, outs) = new TempTimeSeries().GetDataSet();
            var (inputs, target) = Utilities.Layer(ins, outs, dataLayerSize);

            var neat = new Neat()
                .SetBatchSize(10)
                .SetLossFunction(LossFunction.Difference)
                .AddLayer(new LSTM(dataLayerSize, 1, 128, ActivationFunction.Sigmoid));

            // neat = await Evolve(neat, inputs, target, evolveEpochs);
            neat = await Train(neat, inputs, target, maxEpoch, learningRate);
            
            foreach (var (first, second) in inputs.Zip(target))
            {
                Console.WriteLine($"Input {first[0]} Expecting {second[0]} Guess {neat.Forward(first)[0]}");
            }
        }

        private async Task<Neat> Train(Neat neat, List<List<float>> inputs, List<List<float>> targets, int epochs, float learningRate)
        {
            neat.Train(inputs, targets, learningRate, (epoch, loss) =>
            {
                Console.WriteLine($"{epoch} - {loss}");
                return epoch == epochs || Math.Abs(loss) < .01f;
            });

            neat.ResetGenome();
            return neat;
        }

        private async Task<Neat> Evolve(Neat neat, List<List<float>> inputs, List<List<float>> targets, int epochs)
        {
            var best = await new Population<Neat, NeatEnvironment>()
                .Configure(settings =>
                { 
                    settings.Size = 100;
                    settings.DynamicDistance = true;
                    settings.SpeciesTarget = 5;
                    settings.SpeciesDistance = .5;
                    settings.InbreedRate = .001;
                    settings.CrossoverRate = .75;
                    settings.CleanPct = .9;
                    settings.StagnationLimit = 15;
                })
                .SetParentPicker(ParentPickerResolver.Get<Neat>(ParentPicker.BiasedRandom))
                .SetSurvivorPicker(SurvivorPickerResolver.Get<Neat>(SurvivorPicker.Fittest))
                .SetEnvironment(new NeatEnvironment
                {
                    ReactivateRate = .2f,
                    WeightMutateRate = .8f,
                    NewEdgeRate = .14f,
                    NewNodeRate = .14f,
                    EditWeights = .1f,
                    WeightPerturb = 1.2f,
                    ActivationFunctions = new()
                    {
                        ActivationFunction.ReLU
                    }
                })
                .SetSolver(member =>
                {
                    // var meanSquaredError = inputs.Zip(targets)
                    //     .Select(pair => Math.Pow(member.Forward(pair.First)[0] - pair.Second[0], 2))
                    //     .Sum();
                    //
                    // return 1f / inputs.Count * meanSquaredError;

                    var meanSquaredError = inputs.Zip(targets)
                        .Select(pair => Math.Pow(member.Forward(pair.First)[0] - pair.Second[0], 2))
                        .Sum() / inputs.Count;
                    
                    return 1.0 - meanSquaredError;
                })
                .PopulateClone(neat)
                .Train((member, epoch) =>
                {
                    Console.WriteLine($"{member.Fitness} - {epoch}");
                    return epoch == epochs;
                });

            var result = best.Model;
            result.ResetGenome();

            return result;
        }
    }
}