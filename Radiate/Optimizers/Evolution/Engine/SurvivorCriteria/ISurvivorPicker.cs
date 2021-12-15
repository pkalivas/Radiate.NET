﻿
namespace Radiate.Optimizers.Evolution.Engine.SurvivorCriteria;

public interface ISurvivorPicker
{
    List<(Guid memberId, Member<T> member)> Pick<T>(Dictionary<Guid, Member<T>> members, List<Niche> species);
}
