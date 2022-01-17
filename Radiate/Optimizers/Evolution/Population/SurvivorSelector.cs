namespace Radiate.Optimizers.Evolution.Population;

public static class SurvivorSelector
{
    public static List<(Guid memberId, Member<T> member)> Select<T>(Dictionary<Guid, Member<T>> members, List<Niche> species) => 
        species
            .Select(spec =>
            {
                var best = spec.BestMember();
                return (
                    memberId: best,
                    member: members[best]
                );
            })
            .ToList();
}