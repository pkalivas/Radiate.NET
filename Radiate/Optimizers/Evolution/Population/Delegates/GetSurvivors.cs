
namespace Radiate.Optimizers.Evolution.Population.Delegates;

public delegate List<(Guid memberId, Member<T> member)> GetSurvivors<T>(Dictionary<Guid, Member<T>> members, List<Niche> species);
