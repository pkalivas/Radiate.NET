using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Callbacks.Resolver;
using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.TrainingSessions;

public abstract class TrainingSession
{
    private readonly IEnumerable<ITrainingCallback> _callbacks;
    protected readonly List<Epoch> Epochs;

    protected TrainingSession(IEnumerable<ITrainingCallback> callbacks)
    {
        _callbacks = callbacks ?? new List<ITrainingCallback>();
        Epochs = new List<Epoch>();
    }

    public abstract Task<T> Train<T>(TensorTrainSet trainingData, LossFunction lossFunction, Func<Epoch, bool> trainFunc);

    protected async Task CompleteTraining<T>(T model, LossFunction lossFunction)
    {
        foreach (var callback in GetCallbacks<ITrainingCompletedCallback>())
        {
            await callback.CompleteTraining<T>(model, Epochs, lossFunction);
        }
    }
    
    protected List<T> GetCallbacks<T>() => CallbackResolver.Get<T>(_callbacks).ToList();

}