using System;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Net.Data;
using Radiate.NET.Engine;
using Radiate.NET.Engine.Enums;
using Radiate.NET.Engine.ParentalCriteria;
using Radiate.NET.Engine.SurvivorCriteria;
using Radiate.NET.Models.Neat;
using Radiate.NET.Models.Neat.Enums;
using Radiate.NET.Models.Neat.Layers;

namespace Radiate.Net.Examples.Examples
{
    public class EvolveLSTM : IExample
    {
        public async Task Run()
        {
            var (inputs, answers) = new SimpleMemory().GetDataSet();

            var neat = new Neat()
                .AddLayer(new LSTM(1, 1, 10, ActivationFunction.Relu));
            
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
                    WeightPerturb = 1.5f,
                    ActivationFunctions = new()
                    {
                        ActivationFunction.Relu
                    }
                })
                .SetSolver(member =>
                {
                    var meanSquaredError = inputs.Zip(answers)
                        .Select(pair => Math.Pow(member.Forward(pair.First)[0] - pair.Second[0], 2))
                        .Sum() / inputs.Count;
                    
                    return 1.0 - meanSquaredError;
                })
                .PopulateClone(neat)
                .Train((member, epoch) =>
                {
                    Console.WriteLine($"{member.Fitness} - {epoch}");
                    return epoch == 200;
                });
            
            var member = best.Model;
            member.ResetGenome();
            
            foreach (var (point, idx) in inputs.Select((val, idx) => (val, idx)))
            {
                var output = member.Forward(point);
                Console.WriteLine($"Input ({point[0]}) Expecting {answers[idx][0]} Guess {output[0]}");
            }

            Console.WriteLine("\nTesting Memory...");
            Console.WriteLine($"Input {1f} Expecting {0f} Guess {member.Forward(new(){ 1f })[0]}");
            Console.WriteLine($"Input {0f} Expecting {0f} Guess {member.Forward(new(){ 0f })[0]}");
            Console.WriteLine($"Input {0f} Expecting {0f} Guess {member.Forward(new(){ 0f })[0]}");
            Console.WriteLine($"Input {0f} Expecting {1f} Guess {member.Forward(new(){ 0f })[0]}");
        }
    }
}