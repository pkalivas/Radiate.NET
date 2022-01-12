namespace Radiate.Callbacks.Interfaces;

public interface IEpochStartedCallback : ITrainingCallback
{
    void EpochStarted();
}