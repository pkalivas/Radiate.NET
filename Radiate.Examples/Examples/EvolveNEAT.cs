﻿using Radiate.Activations;
using Radiate.Data;
using Radiate.Optimizers;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Neat;
using Radiate.Optimizers.Evolution.Population.Enums;
using Radiate.Optimizers.Evolution.Population.ParentalCriteria;
using Radiate.Optimizers.Evolution.Population.SurvivorCriteria;

namespace Radiate.Examples.Examples;

public class EvolveNEAT : IExample
{
    public async Task Run()
    {
        const int maxEpochs = 500;
        const int populationSize = 200;
        
        var (inputs, answers) = await new SimpleMemory().GetDataSet();

        var networks = Enumerable.Range(0, populationSize).Select(_ => new Neat(1, 1, Activation.ExpSigmoid)).ToList();

        var population = new Population<Neat, NeatEnvironment>(networks)
            .Configure(settings =>
            {
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
                RecurrentNeuronRate = 1f,
                ReactivateRate = .2f,
                WeightMutateRate = .8f,
                NewEdgeRate = .14f,
                NewNodeRate = .14f,
                EditWeights = .1f,
                WeightPerturb = 1.5f,
                ActivationFunctions = new List<Activation>
                {
                    Activation.ExpSigmoid,
                    Activation.ReLU
                }
            })
            .SetFitnessFunction(member =>
            {
                var total = 0.0f;
                foreach (var points in inputs.Zip(answers))
                {
                    var output = member.Forward(points.First);
                    total += (float) Math.Pow((output[0] - points.Second[0]), 2);
                }

                return 1f - (total / inputs.Count);
            });

        var optimizer = new Optimizer<Population<Neat, NeatEnvironment>>(population);
        var member = (await optimizer.Train(epoch =>
        {
            Console.Write($"\r[{epoch.Index}] {epoch.Fitness}");
            return epoch.Index == maxEpochs;
        })).Best;
        
        member.ResetGenome();

        Console.WriteLine();
        foreach (var (point, idx) in inputs.Select((val, idx) => (val, idx)))
        {
            var output = member.Forward(point);
            Console.WriteLine($"Input {point[0]} Expecting {answers[idx][0]} Guess {output[0]}");
        }
        
        Console.WriteLine("\nTesting Memory...");
        Console.WriteLine($"Input {1f} Expecting {0f} Guess {member.Forward(new float[1] { 1 })[0]}");
        Console.WriteLine($"Input {0f} Expecting {0f} Guess {member.Forward(new float[1] { 0 })[0]}");
        Console.WriteLine($"Input {0f} Expecting {0f} Guess {member.Forward(new float[1] { 0 })[0]}");
        Console.WriteLine($"Input {0f} Expecting {1f} Guess {member.Forward(new float[1] { 0 })[0]}");
    }
}

//
// var total = 0.0f;
// foreach (var points in inputs.Zip(answers))
// {
//     var output = member.Forward(points.First);
//     total += (float) Math.Pow((output[0] - points.Second[0]), 2);
// }
//             
// return 4.0f - total;