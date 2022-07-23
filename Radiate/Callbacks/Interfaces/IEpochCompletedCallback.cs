using Radiate.Records;

namespace Radiate.Callbacks.Interfaces;

public interface IEpochCompletedCallback : ITrainingCallback
{
    public void EpochCompleted(Epoch epoch);
}