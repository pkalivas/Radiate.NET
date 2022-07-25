using System.Collections.Concurrent;
using Radiate.Records;

namespace Radiate.Optimizers.Evolution.Managers;

public class StagnationManager
{
    private readonly int _stagnationLimit;
    private readonly ConcurrentDictionary<Guid, (int count, double maxFitness)> _speciesStagnationState;

    public StagnationManager(StagnationControl control)
    {
        var (limit, _, _) = control;
        
        _stagnationLimit = limit;
        _speciesStagnationState = new ConcurrentDictionary<Guid, (int, double)>();
    }
    
    public bool IsStagnant(Guid speciesId) => _speciesStagnationState[speciesId].count == _stagnationLimit;
    public int Stagnation(Guid speciesId) => _speciesStagnationState[speciesId].count;

    public void Update(Guid speciesId, double fitness)
    {
        var (stagnationCount, maxFitness) = _speciesStagnationState.ContainsKey(speciesId)
            ? _speciesStagnationState[speciesId]
            : (0, double.MinValue);
        
        
        if (fitness <= maxFitness)
        {
            _speciesStagnationState[speciesId] = (++stagnationCount, maxFitness);
        }
        else
        {
            _speciesStagnationState[speciesId] = (0, fitness);
        }
    }
}