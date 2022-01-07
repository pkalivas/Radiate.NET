using Radiate.Domain.Callbacks;
using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Callbacks.Resolver;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised;

public class SupervisedTrainingSession
{
    private readonly ISupervised _supervisedModel;
    private readonly IEnumerable<ITrainingCallback> _callbacks;
    private readonly List<Epoch> _epoches;

    public SupervisedTrainingSession(ISupervised supervisedModel, IEnumerable<ITrainingCallback> callbacks)
    {
        _supervisedModel = supervisedModel;
        _callbacks = callbacks;
        _epoches = new List<Epoch>();
    }

    public Epoch Fit(List<Batch> batches, LossFunction lossFunction)
    {
        var predictions = new List<(Prediction, Tensor)>();
        var epochErrors = new List<float>();
        
        foreach (var batch in batches)
        {
            var batchPredictions = FeedBatch(batch);
            
            var batchErrors = batchPredictions
                .Select(pair => lossFunction(pair.prediction.Result, pair.target))
                .ToList();
            
            _supervisedModel.Update(batchErrors, _epoches.Count);
            
            predictions.AddRange(batchPredictions);
            epochErrors.AddRange(batchErrors.Select(err => err.Loss));
        }

        var epoch = Validator.ValidateEpoch(epochErrors, predictions);
        var result = epoch with { Index = _epoches.Count };
        _epoches.Add(epoch);

        foreach (var callback in GetCallbacks<IEpochCompletedCallback>())
        {
            callback.EpochCompleted(result);
        }
        
        return result;
    }

    public void CompleteTraining<T>()
    {
        if (_supervisedModel is T model)
        {
            foreach (var callback in GetCallbacks<ITrainingCompletedCallback>())
            {
                callback.CompleteTraining(model);
            }    
        }
    }

    private List<(Prediction prediction, Tensor target)> FeedBatch(Batch data)
    {
        var (inputs, answers) = data;
        var result = _supervisedModel.Step(inputs, answers);

        var predictions = result.Select(pred => pred.prediction).ToList();
        var targets = result.Select(tar => tar.target).ToList();

        foreach (var callback in GetCallbacks<IBatchCompletedCallback>())
        {
            callback.BatchCompleted(predictions, targets);
        }

        return result;
    }

    private List<T> GetCallbacks<T>() => CallbackResolver.Get<T>(_callbacks).ToList();

}