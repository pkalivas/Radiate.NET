using System.Threading;

namespace Radiate.Optimizers.Evolution;

public class Allele
{
    public static int InnovationCount = RandomGenerator.RandomGenerator.Seed ?? 0;

    protected Random Random { get; }

    protected Allele()
    {
        InnovationCounter.Increment();

        Random = RandomGenerator.RandomGenerator.Seed is null
            ? new Random()
            : new Random(InnovationCounter.Count);
    }

    public static void Reset()
    {
        InnovationCount = RandomGenerator.RandomGenerator.Seed ?? 0;
    }
}