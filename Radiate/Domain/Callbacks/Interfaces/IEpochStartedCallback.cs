namespace Radiate.Domain.Callbacks.Interfaces;

public interface IEpochStartedCallback : ITrainingCallback
{
    void EpochStarted();
}