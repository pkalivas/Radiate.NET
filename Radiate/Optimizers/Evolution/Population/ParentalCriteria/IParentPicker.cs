
namespace Radiate.Optimizers.Evolution.Population.ParentalCriteria;

public interface IParentPicker
{
    (Guid parentOne, Guid parentTwo) Pick<T>(double inbreedRate, List<Niche> species);
}