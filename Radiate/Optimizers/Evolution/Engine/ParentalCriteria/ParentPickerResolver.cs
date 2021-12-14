using System.Collections.Generic;
using Radiate.Optimizers.Evolution.Engine.Delegates;
using Radiate.Optimizers.Evolution.Engine.Enums;

namespace Radiate.Optimizers.Evolution.Engine.ParentalCriteria
{
    public static class ParentPickerResolver
    {
        public static GetParents Get(ParentPicker picker) => picker switch
        {
            ParentPicker.BestInSpecies => new BestInSpecies().Pick,
            ParentPicker.BiasedRandom => new BiasedRandom().Pick,
            _ => throw new KeyNotFoundException($"Parent picker {picker} is not implemented.")
        };
    }

}