using Radiate.Optimizers;
using Radiate.Tensors;

namespace Radiate.Callbacks.Interfaces;

public interface ITrainingCompletedCallback : ITrainingCallback
{
    Task CompleteTraining(Optimizer optimizer, TensorTrainSet tensorSet);
}