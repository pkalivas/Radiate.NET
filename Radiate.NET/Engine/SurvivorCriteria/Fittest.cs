using System;
using System.Collections.Generic;
using System.Linq;

namespace Radiate.NET.Engine.SurvivorCriteria
{
    public class Fittest : ISurvivorPicker
    {
        public List<(Guid memberId, Member<T> member)> Pick<T>(Dictionary<Guid, Member<T>> members, List<Niche> species) => 
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
}