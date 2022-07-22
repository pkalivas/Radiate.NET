using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Records;

namespace Radiate.Callbacks.Interfaces;

public interface IGenerationEvolvedCallback : ITrainingCallback
{
    void GenerationEvolved(Generation generation, PopulationControl populationControl);
}