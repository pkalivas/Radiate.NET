
namespace Radiate.Optimizers.Evolution.Interfaces;

public interface IPopulation
{
    Task<float> Step(int index);
    IGenome Best();
}