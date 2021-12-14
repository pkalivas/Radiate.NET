using System;
using System.Collections.Generic;

namespace Radiate.Optimizers.Evolution.Engine.Delegates
{
    public delegate List<(Guid memberId, Member<Genome> member)> GetSurvivors(Dictionary<Guid, Member<Genome>> members, List<Niche> species);
}