using Radiate.Callbacks;
using Radiate.Callbacks.Interfaces;
using Radiate.Data;
using Radiate.Extensions;
using Radiate.Optimizers;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Genomes.Neat;
using Radiate.Optimizers.Evolution.Info;
using Radiate.Tensors;

namespace Radiate.Examples.Examples;

public class EvolveNEAT : IExample
{
    public async Task Run()
    {
        const int maxEpochs = 500;
        
        var (inputs, answers) = await new SimpleMemory().GetDataSet();

        var pair = new TensorTrainSet(inputs, answers);
        var tensorInputs = pair.InputsToTensorRow();

        var info = new PopulationInfo<Neat>()
            .AddSettings(settings =>
            {
                settings.Size = 100;
                settings.DynamicDistance = true;
                settings.SpeciesTarget = 5;
                settings.SpeciesDistance = .5;
                settings.InbreedRate = .001;
                settings.CrossoverRate = .5;
                settings.StagnationLimit = 15;
                settings.COne = 1;
                settings.CTwo = 1;
                settings.CThree = .03;
            })
            .AddEnvironment(() =>
            {
                var environment = DefaultEnvironments.RecurrentNeatEnvironment;
                environment.InputSize = pair.InputShape.Height;
                environment.OutputSize = pair.OutputShape.Height;

                return environment;
            })
            .AddFitnessFunction(member => DefaultFitnessFunctions.MeanSquaredError(member, tensorInputs));

        var population = new Population<Neat>(info);
        var optimizer = new Optimizer(population, pair, new List<ITrainingCallback>
        {
            new GenerationCallback(),
            new FreeStyleCallback()
        });
        
        await optimizer.Train<Neat>(epoch => epoch.Index == maxEpochs);
    }
}
