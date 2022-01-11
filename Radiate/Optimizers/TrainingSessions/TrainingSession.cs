using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Callbacks.Resolver;
using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

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

    public abstract Task<T> Train<T>(TensorTrainSet trainingData, LossFunction lossFunction, Func<Epoch, bool> trainFunc);
    
    protected List<T> GetCallbacks<T>() => CallbackResolver.Get<T>(_callbacks).ToList();

}