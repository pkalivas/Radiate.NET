using System.Threading;

namespace Radiate.Optimizers.Evolution;

public static class InnovationCounter
{
    private static int _count;

    public static void Init()
    {
        _count = RandomGenerator.RandomGenerator.Seed ?? 0;
    }

    public static int Increment()
    {
        Interlocked.Increment(ref _count);

        return _count;
    }
}