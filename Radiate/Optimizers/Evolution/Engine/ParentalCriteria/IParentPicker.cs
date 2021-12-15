
namespace Radiate.Optimizers.Evolution.Engine.ParentalCriteria;

public interface IParentPicker
{
    (Guid parentOne, Guid parentTwo) Pick(double inbreedRate, List<Niche> species);
}
