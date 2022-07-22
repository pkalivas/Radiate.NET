using System.Threading;

namespace Radiate.RandomGenerator;

public static class RandomGenerator
{
    public static int Seed { get; set; }

    public static Random Next => Seed is 0 ? new Random() : new Random(Seed);
}