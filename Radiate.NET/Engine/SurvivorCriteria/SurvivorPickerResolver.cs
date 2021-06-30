using System.Collections.Generic;
using Radiate.NET.Engine.Delegates;
using Radiate.NET.Engine.Enums;

namespace Radiate.NET.Engine.SurvivorCriteria
{
    public static class SurvivorPickerResolver
    {
        public static GetSurvivors<T> Get<T>(SurvivorPicker picker) => picker switch
        {
            SurvivorPicker.Fittest => new Fittest().Pick,
            _ => throw new KeyNotFoundException($"Survivor picker {picker} is not implemented.")
        };
    }
}