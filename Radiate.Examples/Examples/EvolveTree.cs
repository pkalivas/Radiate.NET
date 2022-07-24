using Radiate.Activations;
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
                settings.SpeciesDistance = .5;
                settings.InbreedRate = .001;
                settings.CrossoverRate = .5;
                settings.StagnationLimit = 15;
            })
            .AddEnvironment(new ForestEnvironment
            {
                InputSize = pair.InputShape.Height,
                OutputCategories = pair.OutputCategoriesList,
                MaxHeight = 20,
                StartHeight = 5,
                NumTrees = 25,
                NodeAddRate = .05f,
                ShuffleRate = .05f,
                NodeType = SeralTreeNodeType.Operator,
                OperatorNodeEnvironment = new OperatorNodeEnvironment
                {
                    SplitValueMutateRate  = .1f,
                    SplitIndexMutateRate = .1f,
                    OutputCategoryMutateRate = .1f,
                    OperatorMutateRate = .05f
                },
                NeuronNodeEnvironment = new NeuronNodeEnvironment
                {
                    SplitIndexMutateRate = .1f,
                    OutputCategoryMutateRate = .1f,
                    WeightMovementRate = 1.5f,
                    WeightMutateRate = .8f,
                    EditWeights = .1f,
                    Activations = new[]
                    {
                        Activation.Linear,
                        Activation.Sigmoid
                    }
                }
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