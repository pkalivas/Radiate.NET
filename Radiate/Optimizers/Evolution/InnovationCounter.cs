using System.Threading;

namespace Radiate.Optimizers.Evolution;

public static class InnovationCounter
{
    public static int Count;

    public static void Init()
    {
        Count = RandomGenerator.RandomGenerator.Seed ?? 0;
    }

    public static int Increment()
    {
        Interlocked.Increment(ref Count);

        return Count;
    }
}