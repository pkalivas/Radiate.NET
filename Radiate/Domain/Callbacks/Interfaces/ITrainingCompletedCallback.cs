using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers;

namespace Radiate.Domain.Callbacks.Interfaces;

public interface ITrainingCompletedCallback : ITrainingCallback
{
    Task CompleteTraining<T>(Optimizer<T> optimizer, List<Epoch> epochs, TensorTrainSet tensorSet) where T : class;
}