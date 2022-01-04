using Radiate.Optimizers.Evolution.Population.Delegates;
using Radiate.Optimizers.Evolution.Population.Enums;

namespace Radiate.Optimizers.Evolution.Population.ParentalCriteria;

public static class ParentPickerResolver
{
    public static GetParents<T> Get<T>(ParentPicker picker) => picker switch
    {
        ParentPicker.BestInSpecies => new BestInSpecies().Pick<T>,
        ParentPicker.BiasedRandom => new BiasedRandom().Pick<T>,
        _ => throw new KeyNotFoundException($"Parent picker {picker} is not implemented.")
    };
}