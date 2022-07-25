using Radiate.Callbacks;
using Radiate.Callbacks.Interfaces;
using Radiate.Losses;
using Radiate.Optimizers;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.TrainingSessions;

public abstract class TrainingSession
{
    private readonly IEnumerable<ITrainingCallback> _callbacks;

    protected TrainingSession(IEnumerable<ITrainingCallback> callbacks)
    {
        _callbacks = callbacks ?? new List<ITrainingCallback>();
    }

    public abstract Task<IOptimizerModel> Train(TensorTrainSet trainingData, Func<Epoch, Task<bool>> trainFunc,
        LossFunction lossFunction);
    
    protected List<T> GetCallbacks<T>() => CallbackResolver.Get<T>(_callbacks).ToList();

    public async Task CompleteTraining(Optimizer optimizer, TensorTrainSet tensorTrainSet)
    {
        foreach (var callback in GetCallbacks<ITrainingCompletedCallback>())
        {
            await callback.CompleteTraining(optimizer, tensorTrainSet);
        }
    }
}