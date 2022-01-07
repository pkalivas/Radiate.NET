using Radiate.Domain.Callbacks.Interfaces;
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

    public Epoch Fit(Tensor[] inputs)
    {
        foreach (var callback in GetCallbacks<IEpochStartedCallback>())
        {
            callback.EpochStarted();
        }

        var loss = _unsupervisedModel.Step(inputs, Epochs.Count);
        var epoch = new Epoch(Epochs.Count + 1, loss);
        Epochs.Add(epoch);
        
        _unsupervisedModel.Update();

        foreach (var callback in GetCallbacks<IEpochCompletedCallback>())
        {
            callback.EpochCompleted(epoch);
        }

        return epoch;
    }
    
    public async Task CompleteTraining<T>(LossFunction lossFunction)
    {
        if (_unsupervisedModel is T model)
        {
            foreach (var callback in GetCallbacks<ITrainingCompletedCallback>())
            {
                await callback.CompleteTraining(model, Epochs, lossFunction);
            }    
        }
    }
}