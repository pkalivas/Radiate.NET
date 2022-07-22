using System.Threading;

namespace Radiate.RandomGenerator;

public static class RandomGenerator
{
    public static int Seed = 0;

    private static readonly ThreadLocal<Random> Random = new ThreadLocal<Random>(() =>
        new Random(Interlocked.Increment(ref Seed)));

    // public static Random Next => Seed is 0 ? new Random() : new Random(Seed);
    public static Random Next => Random.Value;
}