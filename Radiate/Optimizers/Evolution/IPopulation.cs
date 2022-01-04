using Radiate.Domain.Records;

namespace Radiate.Optimizers.Evolution;

public interface IPopulation
{
    Task Evolve(Func<Epoch, bool> trainFunc);
}