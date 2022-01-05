using Radiate.Data;
using Radiate.Domain.Activation;
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
        const int populationSize = 100;
        
        var (inputs, answers) = await new XOR().GetDataSet();

        var networks = new List<Neat>();
        foreach (var _ in Enumerable.Range(0, populationSize))
        {
            networks.Add(new Neat(2, 1, Activation.ExpSigmoid));
        }

        var population = new Population<Neat, NeatEnvironment>(networks)
            .Configure(settings =>
            {
                settings.Size = populationSize;
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
            
                return 4.0f - total;
            });

        var optimizer = new Optimizer<Population<Neat, NeatEnvironment>>(population);
        await optimizer.Train(epoch =>
        {
            Console.WriteLine($"{epoch.Fitness}");
            return epoch.Index == maxEpochs;
        });
        
        var member = optimizer.Model.Best;
        member.ResetGenome();
        
        foreach (var (point, idx) in inputs.Select((val, idx) => (val, idx)))
        {
            var output = member.Forward(point);
            Console.WriteLine($"Input ({point[0]} {point[1]}) output ({output[0]} answer ({answers[idx][0]})");
        }
    }
}