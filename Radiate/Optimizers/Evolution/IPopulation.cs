namespace Radiate.Optimizers.Evolution;

public interface IPopulation
{
    Task Evolve(Func<double, int, bool> trainFunc);
}