using System.Collections.Generic;
using Radiate.NET.Engine.Delegates;
using Radiate.NET.Engine.Enums;

namespace Radiate.NET.Engine.ParentalCriteria
{
    public static class ParentPickerResolver
    {
        public static GetParents<T> Get<T>(ParentPicker picker) => picker switch
        {
            ParentPicker.BestInSpecies => new BestInSpecies().Pick<T>,
            ParentPicker.BiasedRandom => new BiasedRandom().Pick<T>,
            _ => throw new KeyNotFoundException($"Parent picker {picker} is not implemented.")
        };
    }

}