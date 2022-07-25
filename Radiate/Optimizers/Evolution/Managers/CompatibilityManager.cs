using Radiate.Records;

namespace Radiate.Optimizers.Evolution.Managers;

public class CompatibilityManager
{
    private readonly bool _dynamic;
    private readonly int _targetSpecies;

    public CompatibilityManager(CompatibilityControl compatibilityControl)
    {
        var (dynamic, target, distance) = compatibilityControl;
        
        _dynamic = dynamic;
        Distance = distance;
        _targetSpecies = target;
    }

    public double Distance { get; private set; }

    public void Update(int speciesCount)
    {
        if (!_dynamic)
        {
            return;
        }
        
        if (speciesCount < _targetSpecies)
        {
            var currentPrecision = EvolutionConstants.DistancePrecision;
            var newDistance = Distance - currentPrecision;
            while (newDistance <= 0)
            {
                currentPrecision /= EvolutionConstants.DistanceMultiplier;
                newDistance = Distance - currentPrecision;
            }

            Distance = Math.Round(newDistance, 10);
        }
        else if (speciesCount > _targetSpecies)
        {
            var newDistance = Distance + EvolutionConstants.DistancePrecision;
            Distance = Math.Round(newDistance, 10);
        }
    }
}