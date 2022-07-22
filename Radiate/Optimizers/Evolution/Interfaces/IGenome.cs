using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Optimizers.Evolution.Interfaces;

public interface IGenome : IOptimizerModel
{
    public T Crossover<T, TE>(T other, TE environment, double crossoverRate)
        where T: class, IGenome
        where TE: EvolutionEnvironment;

    public Task<double> Distance<T, TE>(T other, TE environment);

    public T CloneGenome<T>() where T : class;

    public void ResetGenome();
}