namespace Radiate.Domain.Callbacks.Interfaces;

public interface ITrainingCompletedCallback
{
    public void CompleteTraining<T>(T model);
}