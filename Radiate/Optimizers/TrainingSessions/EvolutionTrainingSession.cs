using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Evolution;

namespace Radiate.Optimizers.TrainingSessions;

public class EvolutionTrainingSession : TrainingSession
{
    private readonly IPopulation _population;

    public EvolutionTrainingSession(IPopulation population, IEnumerable<ITrainingCallback> callbacks) : base(callbacks)
    {
        _population = population;
    }

    public override async Task<T> Train<T>(TensorTrainSet trainingData, LossFunction lossFunction,
        Func<Epoch, bool> trainFunc)
    {
        while (true)
        {
            var epoch = await Fit();

            if (trainFunc(epoch))
            {
                break;
            }
        }
        
        await CompleteTraining(_population, lossFunction);

        return (T)_population;
    }


    private async Task<Epoch> Fit()
    {
        foreach (var callback in GetCallbacks<IEpochStartedCallback>())
        {
            callback.EpochStarted();
        }
        
        var fitness = await _population.Step();
        var epoch = new Epoch(Epochs.Count + 1, 0f, 0f, 0f, 0f, fitness);
        
        Epochs.Add(epoch);
        
        foreach (var callback in GetCallbacks<IEpochCompletedCallback>())
        {
            callback.EpochCompleted(epoch);
        }

        return epoch;
    }
    
}