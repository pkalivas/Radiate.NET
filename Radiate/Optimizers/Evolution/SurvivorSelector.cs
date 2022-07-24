using System.Collections.Concurrent;

namespace Radiate.Optimizers.Evolution;

public static class SurvivorSelector
{
    public static List<Guid> Select(ConcurrentDictionary<Guid, Niche> species) => 
            species.Values
                .Where(spec => !spec.IsStagnant)
                .Select(spec => spec.BestMember())
                .ToList();
}