using System;
using System.Collections.Generic;

namespace Radiate.NET.Engine.ParentalCriteria
{
    public interface IParentPicker
    {
        (Guid parentOne, Guid parentTwo) Pick<T>(double inbreedRate, List<Niche> species);
    }
}