using Radiate.Optimizers.Evolution;

namespace Radiate.Callbacks.Interfaces;

public interface IGenerationEvolvedCallback : ITrainingCallback
{
    void GenerationEvolved(Generation generation);
}