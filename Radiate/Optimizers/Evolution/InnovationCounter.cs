using System.Threading;

namespace Radiate.Optimizers.Evolution;

public class InnovationCounter
{
    public static int Count;

    public InnovationCounter()
    {
        Count = RandomGenerator.RandomGenerator.Seed ?? 0;
    }

    public static void Increment()
    {
        Interlocked.Increment(ref Count);
    }
}