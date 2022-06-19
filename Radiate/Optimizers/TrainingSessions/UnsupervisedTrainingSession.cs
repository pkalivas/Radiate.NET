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

    public override async Task<IOptimizerModel> Train(TensorTrainSet trainingData, Func<Epoch, Task<bool>> trainFunc, LossFunction lossFunction)
    {
        var index = 0;
        var data = trainingData.TrainingInputs
            .SelectMany(batch => batch.Features.Select(row => row))
            .ToArray();

        while (true)
        {
            var epoch = Fit(index++, data);

            if (await trainFunc(epoch))
            {
                break;
            }
        }
        
        return _unsupervisedModel;
    }


    private Epoch Fit(int index, Tensor[] inputs)
    {
        foreach (var callback in GetCallbacks<IEpochStartedCallback>())
        {
            callback.EpochStarted();
        }

        var loss = _unsupervisedModel.Step(inputs, index);
        var epoch = new Epoch(index, loss);
        
        foreach (var callback in GetCallbacks<IEpochCompletedCallback>())
        {
            callback.EpochCompleted(epoch);
        }

        return epoch;
    }
}