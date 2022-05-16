
using Radiate.Optimizers.Evolution.Environment;

namespace Radiate.Optimizers.Evolution;

public interface IPopulation
{
    Task<float> Step();
    IGenome Best();
}