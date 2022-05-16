using Radiate.Callbacks.Interfaces;
using Radiate.Losses;
using Radiate.Optimizers.Unsupervised;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.TrainingSessions;

public class UnsupervisedTrainingSession : TrainingSession
{
    private readonly IUnsupervised _unsupervisedModel;

    public UnsupervisedTrainingSession(IUnsupervised unsupervised, IEnumerable<ITrainingCallback> callbacks) : base(callbacks)
    {
        _unsupervisedModel = unsupervised;
    }

    public override async Task<T> Train<T>(TensorTrainSet trainingData, LossFunction lossFunction, Func<Epoch, Task<bool>> trainFunc)
    {
        var data = trainingData.TrainingInputs
            .SelectMany(batch => batch.Features.Select(row => row))
            .ToArray();

        while (true)
        {
            var epoch = Fit(data);

            if (await trainFunc(epoch))
            {
                break;
            }
        }
        
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