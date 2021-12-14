using System.Collections.Generic;
using Radiate.Optimizers.Evolution.Engine.Delegates;
using Radiate.Optimizers.Evolution.Engine.Enums;

namespace Radiate.Optimizers.Evolution.Engine.SurvivorCriteria
{
    public static class SurvivorPickerResolver
    {
        public static GetSurvivors Get(SurvivorPicker picker) => picker switch
        {
            SurvivorPicker.Fittest => new Fittest().Pick,
            _ => throw new KeyNotFoundException($"Survivor picker {picker} is not implemented.")
        };
    }
}