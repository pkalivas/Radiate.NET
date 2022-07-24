using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public class DistanceManager
{
    private readonly bool _dynamic;
    private readonly int _targetSpecies;
    private double _distance;

    public DistanceManager(bool dynamic, int target, double distance)
    {
        _dynamic = dynamic;
        _distance = distance;
        _targetSpecies = target;
    }

    public double Distance => _distance;

    public void Update(int speciesCount)
    {
        if (!_dynamic)
        {
            return;
        }
        
        if (speciesCount < _targetSpecies)
        {
            var currentPrecision = EvolutionConstants.DistancePrecision;
            var newDistance = _distance - currentPrecision;
            while (newDistance <= 0)
            {
                currentPrecision /= EvolutionConstants.DistanceMultiplier;
                newDistance = _distance - currentPrecision;
            }

            _distance = Math.Round(newDistance, 10);
        }
        else if (speciesCount > _targetSpecies)
        {
            var newDistance = _distance + EvolutionConstants.DistancePrecision;
            _distance = Math.Round(newDistance, 10);
        }
    }
}