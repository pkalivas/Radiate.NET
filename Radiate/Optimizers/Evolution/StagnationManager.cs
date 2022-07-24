using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public class StagnationManager
{
    private readonly int _stagnationLimit;
    private double _cleanPercent;
    private double _previousFitness;
    private int _stagnationCount;

    public StagnationManager(StagnationControl control)
    {
        var (cleanPercent, limit, count, previousFit) = control;
        
        _cleanPercent = cleanPercent;
        _stagnationLimit = limit;
        _stagnationCount = count;
        _previousFitness = previousFit;
    }

    public bool IsStagnant => _stagnationCount == _stagnationLimit;

    public void Update(double fitness)
    {
        if (Math.Abs(_previousFitness - fitness) < EvolutionConstants.Tolerance)
        {
            _stagnationCount++;
        }
        else
        {
            _stagnationCount = 0;
        }

        _previousFitness = fitness;
    }
    
    
}