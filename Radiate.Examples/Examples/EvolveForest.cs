using Radiate.Callbacks;
using Radiate.Callbacks.Interfaces;
using Radiate.Data;
using Radiate.Optimizers;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Forest;
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

        var features = pair.TrainingInputs.SelectMany(batch => batch.Features);
        var targets = pair.TrainingInputs.SelectMany(batch => batch.Targets);
        
        var info = new PopulationInfo<SeralForest>()
            .AddSettings(settings =>
            {
                settings.Size = 25;
                settings.DynamicDistance = true;
                settings.SpeciesTarget = 5;
                settings.SpeciesDistance = .5;
                settings.InbreedRate = .001;
                settings.CrossoverRate = .5;
                settings.CleanPct = .9;
                settings.StagnationLimit = 15;
            })
            .AddEnvironment(new ForestEnvironment
            {
                InputSize = pair.InputShape.Height,
                OutputCategories = pair.OutputCategoriesList,
                MaxHeight = 35,
                NumTrees = 25,
                NodeAddRate = .05f,
                GutRate = .05f,
                ShuffleRate = .05f,
                SplitValueMutateRate = .08f,
                SplitIndexMutateRate = .08f,
                OutputCategoryMutateRate = 0.1f,
                OperatorMutateRate = .05f
            })
            .AddFitnessFunction(member =>
            {
                var total = 0.0f;
                foreach (var points in features.Zip(targets))
                {
                    var output = member.Predict(points.First);
                    total += (float) Math.Pow((output.Confidence - points.Second[0]), 2);
                }

                return 1f - (total / features.Count());
            });

        var population = new Population<SeralForest>(info);
        var optimizer = new Optimizer(population, pair, new ITrainingCallback[]
        {
            new GenerationCallback(),
            new ModelWriterCallback(),
        });
        
        await optimizer.Train<SeralForest>(epoch => epoch.Index == 100);
        Console.WriteLine($"{optimizer.ValidationScores()}");
    }
}