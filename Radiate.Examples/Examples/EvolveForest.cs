﻿using Radiate.Activations;
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
                settings.DynamicDistance = false;
                settings.SpeciesTarget = 10;
                settings.SpeciesDistance = 5;
                settings.InbreedRate = .005;
                settings.CrossoverRate = .8;
                settings.StagnationLimit = 15;
                settings.COne = 1.0;
                settings.CTwo = 1;
                settings.CThree = 2f;
            })
            .AddEnvironment(() =>
            {
                var environment = DefaultEnvironments.NeuronForest;
                environment.MaxHeight = 7;
                environment.InputSize = pair.InputShape.Height;
                environment.OutputCategories = pair.OutputCategoriesList;
                environment.NeuronNodeEnvironment.FeatureIndexCount = (int)(pair.InputShape.Height * .25f);

                return environment;
            })
            .AddFitnessFunction(member => DefaultFitnessFunctions.MeanSquaredError(member, features, targets));

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