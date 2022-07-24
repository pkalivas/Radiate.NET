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
        
        var info = new PopulationInfo<SeralTree>()
            .AddSettings(settings =>
            {
                settings.Size = 250;
                settings.DynamicDistance = true;
                settings.SpeciesTarget = 5;
                settings.SpeciesDistance = .5;
                settings.InbreedRate = .001;
                settings.CrossoverRate = .5;
                settings.StagnationLimit = 15;
                settings.COne = 1.0;
                settings.CTwo = 1.0;
                settings.CThree = 3.0;
            })
            .AddEnvironment(new ForestEnvironment
            {
                InputSize = pair.InputShape.Height,
                OutputCategories = pair.OutputCategoriesList,
                MaxHeight = 30,
                NumTrees = 25,
                StartHeight = 30,
                NodeAddRate = .05f,
                ShuffleRate = .05f,
                NodeType = SeralTreeNodeType.Neuron,
                OperatorNodeEnvironment = new OperatorNodeEnvironment
                {
                  SplitValueMutateRate  = .1f,
                  SplitIndexMutateRate = .1f,
                  OutputCategoryMutateRate = .1f,
                  OperatorMutateRate = .05f
                },
                NeuronNodeEnvironment = new NeuronNodeEnvironment
                {
                    UseRecurrent = false,
                    SplitIndexMutateRate = .05f,
                    OutputCategoryMutateRate = .05f,
                    WeightMovementRate = 1.1f,
                    WeightMutateRate = .15f,
                    EditWeights = .01f,
                    Activations = new[]
                    {
                        Activation.Sigmoid,
                        Activation.ExpSigmoid,
                        Activation.ReLU
                    }
                }
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

        var population = new Population<SeralTree>(info);
        var optimizer = new Optimizer(population, pair, new ITrainingCallback[]
        {
            new GenerationCallback(),
            new ModelWriterCallback(),
        });
        
        await optimizer.Train<SeralForest>(epoch => epoch.Index == 200);
        Console.WriteLine($"{optimizer.ValidationScores()}");
    }
}