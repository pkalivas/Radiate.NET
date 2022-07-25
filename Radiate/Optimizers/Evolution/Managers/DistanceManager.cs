using System.Collections.Concurrent;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Records;

namespace Radiate.Optimizers.Evolution.Managers;

internal record GenomeDistancePair(Guid One, Guid Two);

public class DistanceManager
{
    private readonly ConcurrentDictionary<GenomeDistancePair, double> _distanceCache;
    private readonly DistanceControl _distanceControl;

    public DistanceManager(DistanceControl distanceControl)
    {
        _distanceCache = new ConcurrentDictionary<GenomeDistancePair, double>();
        _distanceControl = distanceControl;
    }

    public void Clear()
    {
        _distanceCache.Clear();
    }

    public double Distance(Guid oneId, Guid twoId, IGenome oneGenome, IGenome twoGenome)
    {
        var oneKey = new GenomeDistancePair(oneId, twoId);
        var twoKey = new GenomeDistancePair(twoId, oneId);
        if (_distanceCache.TryGetValue(oneKey, out var oneDistance))
        {
            return oneDistance;
        }

        if (_distanceCache.TryGetValue(twoKey, out var twoDistance))
        {
            return twoDistance;
        }

        var distance = oneGenome.Distance(twoGenome, _distanceControl);

        _distanceCache[oneKey] = distance;
        _distanceCache[twoKey] = distance;

        return distance;
    }
}