using Radiate.Callbacks;
using Radiate.Callbacks.Interfaces;
using Radiate.Data;
using Radiate.Losses;
using Radiate.Optimizers;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Genomes.Forest;
using Radiate.Optimizers.Evolution.Info;
using Radiate.Tensors;
using Radiate.Tensors.Enums;

namespace Radiate.Examples.Examples;

public class EvolveForest : IExample
{
    public async Task Run()
    {
        var (rawInputs, rawTargets) = await new BostonHousing().GetDataSet();
        
        var pair = new TensorTrainSet(rawInputs, rawTargets)
            .TransformFeatures(Norm.Standardize)
            .Split();

        var inputs = pair.InputsToTensorRow();
        
        var info = new PopulationInfo<SeralTree>()
            .AddSettings(settings =>
            {
                settings.Size = 250;
                settings.DynamicDistance = true;
                settings.SpeciesTarget = 10;
                settings.SpeciesDistance = 5;
                settings.InbreedRate = .005;
                settings.CrossoverRate = .8;
                settings.StagnationLimit = 15;
                settings.COne = 1.0;
                settings.CTwo = 1;
                settings.CThree = .3f;
            })
            .AddEnvironment(() =>
            {
                var environment = DefaultEnvironments.NeuronForest;
                environment.MaxHeight = 7;
                environment.InputSize = pair.InputShape.Height;
                environment.OutputCategories = pair.OutputCategoriesList;

                return environment;
            })
            .AddFitnessFunction(member => DefaultFitnessFunctions.MeanSquaredError(member, inputs));

        var population = new Population<SeralTree>(info);
        var optimizer = new Optimizer(population, pair, new ITrainingCallback[]
        {
            new GenerationCallback(),
            new ModelWriterCallback(),
            new ShowRegressionCallback()
        });
        
        await optimizer.Train<SeralTree>(epoch => epoch.Index == 200);
        Console.WriteLine($"{optimizer.ValidationScores()}");
    }
}