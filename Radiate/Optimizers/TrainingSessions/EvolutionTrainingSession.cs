using Radiate.Activations;
using Radiate.Callbacks.Interfaces;
using Radiate.Losses;
using Radiate.Optimizers.Evolution;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.TrainingSessions;

public class EvolutionTrainingSession : TrainingSession
{
    private readonly IPopulation _population;

    public EvolutionTrainingSession(IPopulation population, IEnumerable<ITrainingCallback> callbacks) : base(callbacks)
    {
        _population = population;
    }

    public override async Task<T> Train<T>(TensorTrainSet trainingData, LossFunction lossFunction, Func<Epoch, Task<bool>> trainFunc) where T : class
    {
        while (true)
        {
            var epoch = await Fit();

            if (await trainFunc(epoch))
            {
                break;
            }
        }

        var bestMember = _population.Best(); 
        bestMember.ResetGenome();
        return (T)bestMember;
    }

    private async Task<Epoch> Fit()
    {
        foreach (var callback in GetCallbacks<IEpochStartedCallback>())
        {
            callback.EpochStarted();
        }

        var startTime = DateTime.UtcNow;
        var fitness = await _population.Step();
        var endTime = DateTime.UtcNow;
        var epoch = new Epoch(Epochs.Count + 1, 0f, 0f, 0f, 0f, fitness, startTime, endTime);
        
        Epochs.Add(epoch);
        
        foreach (var callback in GetCallbacks<IEpochCompletedCallback>())
        {
            callback.EpochCompleted(epoch);
        }

        return epoch;
    }
    
}