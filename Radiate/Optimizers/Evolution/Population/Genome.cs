
namespace Radiate.Optimizers.Evolution.Population;

public abstract class Genome
{
    public abstract T Crossover<T, TE>(T other, TE environment, double crossoverRate)
        where T: Genome
        where TE: EvolutionEnvironment;

    public abstract Task<double> Distance<T, TE>(T other, TE environment);

    public abstract T CloneGenome<T>() where T: class;

    public abstract void ResetGenome();
}