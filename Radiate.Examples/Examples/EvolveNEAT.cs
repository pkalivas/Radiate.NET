using Radiate.Activations;
using Radiate.Data;
using Radiate.Optimizers;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Neat;

namespace Radiate.Examples.Examples;

public class EvolveNEAT : IExample
{
    public async Task Run()
    {
        const int maxEpochs = 50;
        
        var (inputs, answers) = await new SimpleMemory().GetDataSet();

        var population = new Population<Neat>()
            .AddSettings(settings =>
            {
                settings.Size = 50;
                settings.DynamicDistance = true;
                settings.SpeciesTarget = 5;
                settings.SpeciesDistance = .5;
                settings.InbreedRate = .001;
                settings.CrossoverRate = .5;
                settings.CleanPct = .9;
                settings.StagnationLimit = 15;
            })
            .AddEnvironment(new NeatEnvironment
            {
                InputSize = 1,
                OutputSize = 1,
                RecurrentNeuronRate = .95f,
                ReactivateRate = .2f,
                WeightMutateRate = .8f,
                NewEdgeRate = .14f,
                NewNodeRate = .14f,
                EditWeights = .1f,
                WeightPerturb = 1.5f,
                OutputLayerActivation = Activation.ExpSigmoid,
                ActivationFunctions = new List<Activation>
                {
                    Activation.ExpSigmoid,
                    Activation.ReLU
                }
            })
            .AddFitnessFunction(member =>
            {
                var total = 0.0f;
                foreach (var points in inputs.Zip(answers))
                {
                    var output = member.Forward(points.First);
                    total += (float) Math.Pow((output[0] - points.Second[0]), 2);
                }

                return 1f - (total / inputs.Count);
            });

        var optimizer = new Optimizer(population);
        var pop = await optimizer.Train<Neat>(epoch =>
        {
            Console.Write($"\r[{epoch.Index}] {epoch.Fitness}");
            return epoch.Index == maxEpochs;
        });

        
        Console.WriteLine($"\n{Allele.InnovationCount}");
        Console.WriteLine();
        foreach (var (point, idx) in inputs.Select((val, idx) => (val, idx)))
        {
            var output = pop.Forward(point);
            Console.WriteLine($"Input {point[0]} Expecting {answers[idx][0]} Guess {output[0]}");
        }
        
        Console.WriteLine("\nTesting Memory...");
        Console.WriteLine($"Input {1f} Expecting {0f} Guess {pop.Forward(new float[1] { 1 })[0]}");
        Console.WriteLine($"Input {0f} Expecting {0f} Guess {pop.Forward(new float[1] { 0 })[0]}");
        Console.WriteLine($"Input {0f} Expecting {0f} Guess {pop.Forward(new float[1] { 0 })[0]}");
        Console.WriteLine($"Input {0f} Expecting {1f} Guess {pop.Forward(new float[1] { 0 })[0]}");
    }
}
