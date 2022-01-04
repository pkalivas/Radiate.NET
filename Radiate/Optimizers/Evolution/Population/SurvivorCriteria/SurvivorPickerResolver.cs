using Radiate.Optimizers.Evolution.Population.Delegates;
using Radiate.Optimizers.Evolution.Population.Enums;

namespace Radiate.Optimizers.Evolution.Population.SurvivorCriteria;

public static class SurvivorPickerResolver
{
    public static GetSurvivors<T> Get<T>(SurvivorPicker picker) => picker switch
    {
        SurvivorPicker.Fittest => new Fittest().Pick,
        _ => throw new KeyNotFoundException($"Survivor picker {picker} is not implemented.")
    };
}

