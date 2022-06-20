using Radiate.Callbacks;
using Radiate.Callbacks.Interfaces;
using Radiate.Data;
using Radiate.Optimizers;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Forest;
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
        
        var population = new Population<SeralTree>()
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
                MaxHeight = 20,
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
                var predictions = features
                    .Select(input => member.Predict(input))
                    .ToList();
                
                var score = predictions.Zip(targets)
                    .Sum(pair => pair.First.Classification == (int)pair.Second.Max() ? 1f : 0f);
                
                return score / (float)predictions.Count;
            });

        var optimizer = new Optimizer(population, pair, new ITrainingCallback[]
        {
            new ConfusionMatrixCallback(),
            new ModelWriterCallback(),
        });
        
        await optimizer.Train<SeralTree>(epoch => epoch.Index == 500);
        Console.WriteLine($"{optimizer.ValidationScores()}");
    }
}