using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Callbacks.Resolver;
using Radiate.Domain.Records;

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

    protected List<T> GetCallbacks<T>() => CallbackResolver.Get<T>(_callbacks).ToList();

}