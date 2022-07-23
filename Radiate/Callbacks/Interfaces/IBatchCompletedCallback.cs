using Radiate.Records;

namespace Radiate.Callbacks.Interfaces;

public interface IBatchCompletedCallback : ITrainingCallback
{
    public void BatchCompleted(List<Step> steps);
}