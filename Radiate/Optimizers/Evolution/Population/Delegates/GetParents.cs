
namespace Radiate.Optimizers.Evolution.Population.Delegates;

public delegate (Guid parentOne, Guid parentTwo) GetParents<T>(double inbreedRate, List<Niche> species);