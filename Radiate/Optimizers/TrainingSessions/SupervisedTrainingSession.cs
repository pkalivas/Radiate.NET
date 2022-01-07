﻿using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Supervised;

namespace Radiate.Optimizers.TrainingSessions;

public class SupervisedTrainingSession : TrainingSession
{
    private readonly ISupervised _supervisedModel;
    
    public SupervisedTrainingSession(ISupervised supervisedModel, IEnumerable<ITrainingCallback> callbacks) : base(callbacks)
    {
        _supervisedModel = supervisedModel;
    }

    public Epoch Fit(List<Batch> batches, LossFunction lossFunction)
    {
        foreach (var callback in GetCallbacks<IEpochStartedCallback>())
        {
            callback.EpochStarted();
        }
        
        var predictions = new List<(Prediction, Tensor)>();
        var epochErrors = new List<float>();
        
        foreach (var batch in batches)
        {
            var batchPredictions = FeedBatch(batch);
            
            var batchErrors = batchPredictions
                .Select(pair => lossFunction(pair.prediction.Result, pair.target))
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

    public async Task CompleteTraining<T>(LossFunction lossFunction)
    {
        if (_supervisedModel is T model)
        {
            foreach (var callback in GetCallbacks<ITrainingCompletedCallback>())
            {
                await callback.CompleteTraining(model, Epochs, lossFunction);
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


}