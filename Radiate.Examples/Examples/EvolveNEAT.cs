using Radiate.Activations;
using Radiate.Callbacks;
using Radiate.Callbacks.Interfaces;
using Radiate.Data;
using Radiate.Extensions;
using Radiate.Optimizers;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Forest;
using Radiate.Optimizers.Evolution.Info;
using Radiate.Optimizers.Evolution.Neat;
using Radiate.Tensors;

namespace Radiate.Examples.Examples;

public class EvolveNEAT : IExample
{
    public async Task Run()
    {
        const int maxEpochs = 500;
        
        var (inputs, answers) = await new SimpleMemory().GetDataSet();

        var pair = new TensorTrainSet(inputs, answers);
        var features = pair.TrainingInputs.SelectMany(batch => batch.Features);
        var targets = pair.TrainingInputs.SelectMany(batch => batch.Targets);

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
            .AddFitnessFunction(member => DefaultFitnessFunctions.MeanSquaredError(member, features, targets));

        var population = new Population<Neat>(info);
        var optimizer = new Optimizer(population, new List<ITrainingCallback>
        {
            new GenerationCallback()
        });
        
        var model = await optimizer.Train<Neat>(epoch => epoch.Index == maxEpochs);
        
        Console.WriteLine();
        foreach (var (point, answer) in features.Zip(targets))
        {
            var output = model.Predict(point);
            Console.WriteLine($"Input {point.Max()} Expecting {answer.Max()} Guess {output.Confidence}");
        }
        
        Console.WriteLine("\nTesting Memory...");
        Console.WriteLine($"Input {1f} Expecting {0f} Guess {model.Predict(new float[1] { 1 }.ToTensor())}");
        Console.WriteLine($"Input {0f} Expecting {0f} Guess {model.Predict(new float[1] { 0 }.ToTensor())}");
        Console.WriteLine($"Input {0f} Expecting {0f} Guess {model.Predict(new float[1] { 0 }.ToTensor())}");
        Console.WriteLine($"Input {0f} Expecting {1f} Guess {model.Predict(new float[1] { 0 }.ToTensor())}");
    }
}
