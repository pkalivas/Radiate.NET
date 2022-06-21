﻿using Radiate.Callbacks.Interfaces;
using Radiate.Losses;
using Radiate.Optimizers;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Environment;
using Radiate.Optimizers.Evolution.Forest;
using Radiate.Optimizers.Evolution.Neat;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.TrainingSessions;

public class EvolutionTrainingSession : TrainingSession
{
    private readonly IPopulation _population;

    public EvolutionTrainingSession(IPopulation population, IEnumerable<ITrainingCallback> callbacks) : base(callbacks)
    {
        _population = population;
    }

    public EvolutionTrainingSession(IGenome genome, IEnumerable<ITrainingCallback> callbacks) : base(callbacks)
    {
        _population = genome switch
        {
            SeralTree tree => new Population<SeralTree>(tree),
            Neat neat => new Population<Neat>(neat),
            SeralForest forest => new Population<SeralForest>(forest),
            _ => throw new Exception("Cannot create evolution training session")
        };
    }

    public override async Task<IOptimizerModel> Train(TensorTrainSet trainingData, Func<Epoch, Task<bool>> trainFunc, LossFunction lossFunction)
    {
        var index = 0;
        while (true)
        {
            var epoch = await Fit(index++);

            if (await trainFunc(epoch))
            {
                break;
            }
        }

        var bestMember = _population.Best(); 
        bestMember.ResetGenome();
        return bestMember as IOptimizerModel;
    }

    private async Task<Epoch> Fit(int index)
    {
        foreach (var callback in GetCallbacks<IEpochStartedCallback>())
        {
            callback.EpochStarted();
        }

        var startTime = DateTime.UtcNow;
        var fitness = await _population.Step();
        var endTime = DateTime.UtcNow;
        var epoch = new Epoch(index, 0f, 0f, 0f, 0f, fitness, startTime, endTime);
        
        foreach (var callback in GetCallbacks<IEpochCompletedCallback>())
        {
            callback.EpochCompleted(epoch);
        }

        return epoch;
    }
    
}