using System;
using System.Collections.Generic;

namespace Radiate.NET.Engine.SurvivorCriteria
{
    public interface ISurvivorPicker
    {
        List<(Guid memberId, Member<T> member)> Pick<T>(Dictionary<Guid, Member<T>> members, List<Niche> species);
    }
}