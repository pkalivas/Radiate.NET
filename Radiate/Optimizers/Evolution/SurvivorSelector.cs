namespace Radiate.Optimizers.Evolution;

public static class SurvivorSelector
{
    public static List<(Guid memberId, Member member)> Select(Dictionary<Guid, Member> members, List<Niche> species) => 
        species
            .Where(spec => !spec.IsStagnant)
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