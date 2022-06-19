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