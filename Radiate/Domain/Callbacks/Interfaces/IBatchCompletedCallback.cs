using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Callbacks.Interfaces;

public interface IBatchCompletedCallback : ITrainingCallback
{
    public void BatchCompleted(List<Prediction> predictions, List<Tensor> targets);
}