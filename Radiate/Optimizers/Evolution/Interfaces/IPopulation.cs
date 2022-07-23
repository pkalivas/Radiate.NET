
namespace Radiate.Optimizers.Evolution.Interfaces;

public interface IPopulation : IOptimizerModel
{
    Task<Generation> Evolve(int index);
    float PassDown();
    IGenome Best();
}