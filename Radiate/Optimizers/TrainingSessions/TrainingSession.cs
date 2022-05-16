﻿using Radiate.Callbacks.Interfaces;
using Radiate.Callbacks.Resolver;
using Radiate.Losses;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.TrainingSessions;

public abstract class TrainingSession
{
    private readonly IEnumerable<ITrainingCallback> _callbacks;
    public readonly List<Epoch> Epochs;

    protected TrainingSession(IEnumerable<ITrainingCallback> callbacks)
    {
        _callbacks = callbacks;
        Epochs = new List<Epoch>();
    }

    public abstract Task<T> Train<T>(TensorTrainSet trainingData, LossFunction lossFunction, Func<Epoch, Task<bool>> trainFunc) where T : class;
    
    protected List<T> GetCallbacks<T>() => CallbackResolver.Get<T>(_callbacks).ToList();

}