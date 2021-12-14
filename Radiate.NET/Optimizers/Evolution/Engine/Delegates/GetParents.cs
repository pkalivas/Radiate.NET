using System;
using System.Collections.Generic;

namespace Radiate.NET.Optimizers.Evolution.Engine.Delegates
{
    public delegate (Guid parentOne, Guid parentTwo) GetParents(double inbreedRate, List<Niche> species);
}