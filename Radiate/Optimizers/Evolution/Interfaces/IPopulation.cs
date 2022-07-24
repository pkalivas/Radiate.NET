
namespace Radiate.Optimizers.Evolution.Interfaces;

public interface IPopulation : IOptimizerModel
{
    Task<GenerationReport> Evolve();
    float PassDown();
    IGenome Best();
}