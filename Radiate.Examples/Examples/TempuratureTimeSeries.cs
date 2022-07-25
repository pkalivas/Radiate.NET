using Radiate.Callbacks;
using Radiate.Callbacks.Interfaces;
using Radiate.Data;
using Radiate.Examples.DefaultSettings;
using Radiate.Optimizers;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Genomes.Neat;
using Radiate.Optimizers.Evolution.Info;
using Radiate.Tensors;
using Radiate.Tensors.Enums;

namespace Radiate.Examples.Examples;

public class TempuratureTimeSeries : IExample
{
    public async Task Run()
    {
        const int maxEpochs = 500;
        
        var (inputs, answers) = await new TempTimeSeries(500).GetDataSet();

        var pair = new TensorTrainSet(inputs, answers)
            .TransformFeatures(Norm.Normalize)
            .TransformTargets(Norm.Normalize)
            .LookForward(1)
            .Split()
            .Layer(5);

        var tensorInputs = pair.InputsToTensorRow();
        
        var info = new PopulationInfo<Neat>()
            .AddSettings(settings =>
            {
                settings.Size = 100;
                settings.DynamicDistance = true;
                settings.SpeciesTarget = 10;
                settings.SpeciesDistance = .5;
                settings.InbreedRate = .001;
                settings.CrossoverRate = .5;
                settings.StagnationLimit = 15;
                settings.COne = 1;
                settings.CTwo = 1;
                settings.CThree = .003;
            })
            .AddEnvironment(() =>
            {
                var environment = DefaultEnvironments.RecurrentNeatEnvironment;
                environment.InputSize = pair.InputShape.Height;
                environment.OutputSize = 1;

                return environment;
            })
            .AddFitnessFunction(member => DefaultFitnessFunctions.MeanSquaredError(member, tensorInputs));

        var population = new Population<Neat>(info);
        var optimizer = new Optimizer(population, pair, new List<ITrainingCallback>
        {
            new GenerationCallback(),
            new PredictionCsvWriterCallback()
        });
        
        await optimizer.Train<Neat>(epoch => epoch.Index == maxEpochs);
    }
}