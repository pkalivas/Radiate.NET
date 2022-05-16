using Radiate.Callbacks.Interfaces;
using Radiate.Losses;
using Radiate.Optimizers.Supervised;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.TrainingSessions;

public class SupervisedTrainingSession : TrainingSession
{
    private readonly ISupervised _supervisedModel;
    
    public SupervisedTrainingSession(ISupervised supervisedModel, IEnumerable<ITrainingCallback> callbacks) : base(callbacks)
    {
        _supervisedModel = supervisedModel;
    }

    public override async Task<T> Train<T>(TensorTrainSet trainingData, LossFunction lossFunction, Func<Epoch, Task<bool>> trainFunc)
    {
        var batches = trainingData.TrainingInputs;
        
        while (true)
        {
            var epoch = Fit(batches, lossFunction);

            if (await trainFunc(epoch))
            {
                break;
            }
        }
        
        return (T) _supervisedModel;
    }

    private Epoch Fit(List<Batch> batches, LossFunction lossFunction)
    {
        foreach (var callback in GetCallbacks<IEpochStartedCallback>())
        {
            callback.EpochStarted();
        }

        var predictions = new List<Step>();
        var epochErrors = new List<float>();
        
        foreach (var batch in batches)
        {
            var batchPredictions = FeedBatch(batch);
            
            var batchErrors = batchPredictions
                .Select(pair => lossFunction(pair.Prediction.Result, pair.Target))
                .ToList();
            
            _supervisedModel.Update(batchErrors, Epochs.Count);
            
            predictions.AddRange(batchPredictions);
            epochErrors.AddRange(batchErrors.Select(err => err.Loss));
        }

        var epoch = Validator.ValidateEpoch(epochErrors, predictions) with { Index = Epochs.Count + 1 };
        Epochs.Add(epoch);

        foreach (var callback in GetCallbacks<IEpochCompletedCallback>())
        {
            callback.EpochCompleted(epoch);
        }
        
        return epoch;
    }

    private List<Step> FeedBatch(Batch data)
    {
        var (inputs, answers) = data;
        var result = _supervisedModel.Step(inputs, answers);
        
        foreach (var callback in GetCallbacks<IBatchCompletedCallback>())
        {
            callback.BatchCompleted(result);
        }

        return result;
    }
}