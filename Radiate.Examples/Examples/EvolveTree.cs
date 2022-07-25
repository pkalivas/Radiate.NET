using Radiate.Callbacks;
using Radiate.Callbacks.Interfaces;
using Radiate.Data;
using Radiate.Examples.DefaultSettings;
using Radiate.Optimizers;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Genomes.Forest;
using Radiate.Optimizers.Evolution.Info;
using Radiate.Tensors;
using Radiate.Tensors.Enums;

namespace Radiate.Examples.Examples;

public class EvolveTree : IExample
{
    public async Task Run()
    {
        var (rawInputs, rawLabels) = await new BreastCancer().GetDataSet();

        var pair = new TensorTrainSet(rawInputs, rawLabels)
            .TransformFeatures(Norm.Standardize)
            .Split()
            .Shuffle();

        var features = pair.TrainingInputs.SelectMany(batch => batch.Features);
        var targets = pair.TrainingInputs.SelectMany(batch => batch.Targets);
        
        var info = new PopulationInfo<SeralTree>()
            .AddSettings(settings =>
            {
                settings.Size = 100;
                settings.DynamicDistance = true;
                settings.SpeciesTarget = 5;
                settings.SpeciesDistance = 3;
                settings.InbreedRate = .001;
                settings.CrossoverRate = .75;
                settings.StagnationLimit = 15;
                settings.CThree = 2f;
            })
            .AddEnvironment(() =>
            {
                var environment = DefaultEnvironments.OperatorNodeForest;
                environment.InputSize = pair.InputShape.Height;
                environment.OutputCategories = pair.OutputCategoriesList;

                return environment;
            })
            .AddFitnessFunction(member =>
            {
                var score = features.Zip(targets)
                    .Sum(pair => member.Predict(pair.First).Classification == (int)pair.Second.Max() ? 1f : 0f);
                
                return score / (float)features.Count();
            });

        var population = new Population<SeralTree>(info);
        var optimizer = new Optimizer(population, pair, new ITrainingCallback[]
        {
            new GenerationCallback(),
            new ConfusionMatrixCallback(),
            new ModelWriterCallback(),
        });
        
        await optimizer.Train<SeralTree>(epoch => epoch.Index == 500);
        Console.WriteLine($"{optimizer.ValidationScores()}");
    }
}