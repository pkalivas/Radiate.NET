using Radiate.Domain.Records;

namespace Radiate.Domain.Callbacks.Interfaces;

public interface IEpochCompletedCallback : ITrainingCallback
{
    public void EpochCompleted(Epoch epoch);
}