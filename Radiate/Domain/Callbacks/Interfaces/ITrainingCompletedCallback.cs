using Radiate.Domain.Loss;
using Radiate.Domain.Records;

namespace Radiate.Domain.Callbacks.Interfaces;

public interface ITrainingCompletedCallback : ITrainingCallback
{
    Task CompleteTraining<T>(T model, List<Epoch> epochs, LossFunction lossFunction);
}