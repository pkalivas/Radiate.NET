
namespace Radiate.Optimizers.Evolution.Interfaces;

public interface IPopulation : IOptimizerModel
{
    Task<Generation> Evolve();
    float PassDown();
    IGenome Best();
}