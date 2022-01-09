﻿using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Unsupervised;

namespace Radiate.Optimizers.TrainingSessions;

public class UnsupervisedTrainingSession : TrainingSession
{
    private readonly IUnsupervised _unsupervisedModel;

    public UnsupervisedTrainingSession(IUnsupervised unsupervised, IEnumerable<ITrainingCallback> callbacks) : base(callbacks)
    {
        _unsupervisedModel = unsupervised;
    }

    public override async Task<T> Train<T>(TensorTrainSet trainingData, LossFunction lossFunction,
        Func<Epoch, bool> trainFunc)
    {
        var data = trainingData.TrainingFeatureInputs
            .SelectMany(batch => batch.Features.Select(row => row))
            .ToArray();

        while (true)
        {
            var epoch = Fit(data);

            if (trainFunc(epoch))
            {
                break;
            }
        }

        await CompleteTraining(_unsupervisedModel, lossFunction);

        return (T)_unsupervisedModel;
    }


    private Epoch Fit(Tensor[] inputs)
    {
        foreach (var callback in GetCallbacks<IEpochStartedCallback>())
        {
            callback.EpochStarted();
        }

        var loss = _unsupervisedModel.Step(inputs, Epochs.Count);
        var epoch = new Epoch(Epochs.Count + 1, loss);
        Epochs.Add(epoch);
        
        foreach (var callback in GetCallbacks<IEpochCompletedCallback>())
        {
            callback.EpochCompleted(epoch);
        }

        return epoch;
    }
}