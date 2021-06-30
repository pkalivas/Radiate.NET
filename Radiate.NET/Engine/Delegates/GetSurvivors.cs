using System;
using System.Collections.Generic;

namespace Radiate.NET.Engine.Delegates
{
    public delegate List<(Guid memberId, Member<T> member)> GetSurvivors<T>(Dictionary<Guid, Member<T>> members, List<Niche> species);
}