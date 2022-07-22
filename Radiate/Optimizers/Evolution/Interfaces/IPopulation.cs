
namespace Radiate.Optimizers.Evolution.Interfaces;

public interface IPopulation
{
    Task<Generation> Evolve(int index);
    float PassDown();
    IGenome Best();
}