using System;
using System.Collections.Generic;
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
    public class EvolveDense : IExample
    {
        public async Task Run()
        {
            var (inputs, answers) = new XOR().GetDataSet();

            var neat = new Neat()
                .AddLayer(new Dense(2, 1, ActivationFunction.Sigmoid));
            
            var best = await new Population<Neat, NeatEnvironment>()
                .Configure(settings =>
                { 
                    settings.Size = 100;
                    settings.DynamicDistance = true;
                    settings.SpeciesTarget = 5;
                    settings.SpeciesDistance = .5;
                    settings.InbreedRate = .001;
                    settings.CrossoverRate = .5;
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
                    ActivationFunctions = new List<ActivationFunction>
                    {
                        ActivationFunction.Sigmoid,
                        ActivationFunction.ReLU
                    }
                })
                .SetSolver(member =>
                {
                    var total = 0.0;
                    foreach (var points in inputs.Zip(answers))
                    {
                        var output = member.Forward(points.First);
                        total += Math.Pow((output[0] - points.Second[0]), 2);
                    }
            
                    return 4.0 - total;
                })
                .PopulateClone(neat)
                .Train((member, epoch) =>
                {
                    Console.WriteLine($"{member.Fitness} - {epoch}");
                    return epoch == 250;
                });
            
            var member = best.Model;
            member.ResetGenome();
            
            foreach (var (point, idx) in inputs.Select((val, idx) => (val, idx)))
            {
                var output = member.Forward(point);
                Console.WriteLine($"Input ({point[0]} {point[1]}) output ({output[0]} answer ({answers[idx][0]})");
            }
        }
    }
}