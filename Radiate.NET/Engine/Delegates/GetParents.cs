using System;
using System.Collections.Generic;

namespace Radiate.NET.Engine.Delegates
{
    public delegate (Guid parentOne, Guid parentTwo) GetParents<T>(double inbreedRate, List<Niche> species);
}