
namespace Radiate.Optimizers.Evolution.Interfaces;

public interface IPopulation
{
    Task<float> Step();
    IGenome Best();
}