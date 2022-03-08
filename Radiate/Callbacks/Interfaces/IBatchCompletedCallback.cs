using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Callbacks.Interfaces;

public interface IBatchCompletedCallback : ITrainingCallback
{
    public void BatchCompleted(List<Step> steps);
}