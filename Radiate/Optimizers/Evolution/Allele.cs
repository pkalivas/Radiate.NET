using System.Threading;

namespace Radiate.Optimizers.Evolution;

public class Allele
{
    protected int InnovationId { get; }
    protected Random Random { get; }

    protected Allele()
    {
        var count = InnovationCounter.Increment();

        InnovationId = count;
        Random = RandomGenerator.RandomGenerator.Seed is null
            ? new Random()
            : new Random(count);
    }
}