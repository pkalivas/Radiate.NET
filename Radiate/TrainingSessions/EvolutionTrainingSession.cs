using Radiate.Callbacks.Interfaces;
using Radiate.Losses;
using Radiate.Optimizers;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Genomes.Forest;
using Radiate.Optimizers.Evolution.Genomes.Neat;
using Radiate.Optimizers.Evolution.Interfaces;
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
        return bestMember;
    }

    private async Task<Epoch> Fit(int index)
    {
        foreach (var callback in GetCallbacks<IEpochStartedCallback>())
        {
            callback.EpochStarted();
        }

        var startTime = DateTime.UtcNow;
        var generationReport = await _population.Evolve();

        foreach (var callback in GetCallbacks<IGenerationEvolvedCallback>())
        {
            callback.GenerationEvolved(index, generationReport);
        }
        
        var fitness = _population.PassDown();
        var epoch = new Epoch(index)
        {
            Fitness = fitness,
            StartTime = startTime,
            EndTime = DateTime.Now
        };
        
        foreach (var callback in GetCallbacks<IEpochCompletedCallback>())
        {
            callback.EpochCompleted(epoch);
        }

        return epoch;
    }
    
}