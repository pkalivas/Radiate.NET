using System.Threading;

namespace Radiate.RandomGenerator;

public static class RandomGenerator
{
    private static Random _local;
    
    private static readonly ThreadLocal<Random> RandomPool = new(() => new Random(Seed!.Value));

    public static int? Seed { get; set; }

    public static Random Next => GetCache();
    
    private static Random GetCache()
    {
        if (_local is null)
        {
            _local = Seed is null
                ? Random.Shared 
                : RandomPool.Value;
        }

        return _local;
    }
    
}