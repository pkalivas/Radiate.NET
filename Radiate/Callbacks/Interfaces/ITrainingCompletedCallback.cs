using Radiate.Optimizers;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Callbacks.Interfaces;

public interface ITrainingCompletedCallback : ITrainingCallback
{
    Task CompleteTraining<T>(Optimizer<T> optimizer, List<Epoch> epochs, TensorTrainSet tensorSet) where T : class;
}