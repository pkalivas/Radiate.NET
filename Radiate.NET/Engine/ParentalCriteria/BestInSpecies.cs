using System;
using System.Collections.Generic;

namespace Radiate.NET.Engine.ParentalCriteria
{
    public class BestInSpecies : IParentPicker
    {
        public (Guid parentOne, Guid parentTwo) Pick<T>(double inbreedRate, List<Niche> species)
        {
            var random = new Random();

            var speciesOne = species[random.Next(species.Count)];
            var speciesTwo = species[random.Next(species.Count)];

            return (speciesOne.BestMember(), speciesTwo.BestMember());
        }

    }
}