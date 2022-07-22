using System.Threading;

namespace Radiate.RandomGenerator;

public static class RandomGenerator
{
    private static readonly ThreadLocal<Random> RandomPool = new(() => Seed is null 
        ? new Random() 
        : new Random(Seed!.Value));

    public static int? Seed { get; set; }

    public static Random Next => Seed is null 
        ? new Random() 
        : new Random(Seed!.Value);

    public static Random Global => RandomPool.Value;
}