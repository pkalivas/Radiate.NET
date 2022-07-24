using Radiate.Activations;
using Radiate.Callbacks;
using Radiate.Callbacks.Interfaces;
using Radiate.Data;
using Radiate.Gradients;
using Radiate.Losses;
using Radiate.Optimizers;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Info;
using Radiate.Optimizers.Evolution.Neat;
using Radiate.Optimizers.Supervised.Perceptrons;
using Radiate.Optimizers.Supervised.Perceptrons.Info;
using Radiate.Tensors;
using Radiate.Tensors.Enums;

namespace Radiate.Examples.Examples;

public class TempuratureTimeSeries : IExample
{
    public async Task Run()
    {
        const int maxEpochs = 500;
        
        var (inputs, answers) = await new TempTimeSeries(500).GetDataSet();

        inputs = inputs.Take(inputs.Count - 1).ToList();
        answers = answers.Skip(1).ToList();
        var pair = new TensorTrainSet(inputs, answers)
            .TransformFeatures(Norm.Normalize)
            .TransformTargets(Norm.Normalize)
            .Split()
            .Layer(5);
            
        var features = pair.TrainingInputs.SelectMany(batch => batch.Features);
        var targets = pair.TrainingInputs.SelectMany(batch => batch.Targets);
        
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
                settings.CThree = 3;
            })
            .AddEnvironment(new NeatEnvironment
            {
                InputSize = pair.InputShape.Height,
                OutputSize = pair.OutputShape.Height,
                RecurrentNeuronRate = .95f,
                ReactivateRate = .2f,
                WeightMutateRate = .8f,
                NewEdgeRate = .14f,
                NewNodeRate = .14f,
                EditWeights = .1f,
                WeightPerturb = 1.5f,
                OutputLayerActivation = Activation.ExpSigmoid,
                ActivationFunctions = new List<Activation>
                {
                    Activation.ExpSigmoid,
                    Activation.ReLU
                }
            })
            .AddFitnessFunction(member =>
            {
                var total = 0.0f;
                foreach (var points in features.Zip(targets))
                {
                    var output = member.Predict(points.First);
                    total += (float)Math.Pow((output.Confidence - points.Second[0]), 2);
                }

                return 1f - (total / inputs.Count);
            });

        var population = new Population<Neat>(info);
        var optimizer = new Optimizer(population, pair, new List<ITrainingCallback>
        {
            new GenerationCallback(),
            new PredictionCsvWriterCallback()
        });
        
        await optimizer.Train<Neat>(epoch => epoch.Index == maxEpochs);
    }
}