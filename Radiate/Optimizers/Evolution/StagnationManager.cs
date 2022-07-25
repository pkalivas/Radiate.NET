using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public class StagnationManager
{
    private readonly int _stagnationLimit;
    private double _maxFitness;
    private int _stagnationCount;

    public StagnationManager(StagnationControl control)
    {
        var (limit, count, maxFitness) = control;
        
        _stagnationLimit = limit;
        _stagnationCount = count;
        _maxFitness = maxFitness;
    }

    public StagnationManager(StagnationManager manager)
    {
        _stagnationLimit = manager._stagnationLimit;
        _stagnationCount = manager._stagnationCount;
        _maxFitness = manager._maxFitness;
    }

    public bool IsStagnant => _stagnationCount == _stagnationLimit;
    public int Stagnation => _stagnationCount;

    public void Update(double fitness)
    {
        if (fitness <= _maxFitness)
        {
            _stagnationCount++;
        }
        else
        {
            _maxFitness = fitness;
            _stagnationCount = 0;
        }
    }
    
    
}