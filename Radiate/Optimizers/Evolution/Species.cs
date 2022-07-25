
using System.Collections.Concurrent;
using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public class Species
{
    private readonly Guid _speciesId;
    private readonly int _age;
    private readonly int _innovationId;
    private readonly StagnationManager _stagnationManager;
    private readonly ConcurrentDictionary<Guid, double> _memberFitness;
    
    public double Fitness { get; private set; }
    public Gene Mascot { get; }

    public Species(Guid speciesId, Gene mascot, StagnationControl stagnationControl)
    {
        var (memberId, memberFitness, _) = mascot;

        Mascot = mascot;
        _speciesId = speciesId;
        _innovationId = InnovationCounter.Increment();
        _stagnationManager = new StagnationManager(stagnationControl);
        _memberFitness = new ConcurrentDictionary<Guid, double>(new[] { new KeyValuePair<Guid, double>(memberId, memberFitness) });
    }

    public Species(Species other, Gene mascot)
    {
        var (mascotId, mascotFitness, _) = mascot;

        Mascot = mascot;
        _speciesId = other._speciesId;
        _age = other._age + 1;
        _innovationId = other._innovationId;
        _stagnationManager = new StagnationManager(other._stagnationManager);
        _memberFitness = new ConcurrentDictionary<Guid, double>(new[] { new KeyValuePair<Guid, double>(mascotId, mascotFitness) });
    }

    public bool IsStagnant => _stagnationManager.IsStagnant;
    
    public IEnumerable<(Guid GenomeId, double Fitness)> Members =>
        _memberFitness.Select(pair => (pair.Key, pair.Value));
    
    public void AddMember(SpeciesMember member)
    {
        _memberFitness[member.GenomeId] = member.Fitness;
    }

    public Guid BestMember() => _memberFitness.MaxBy(mem => mem.Value).Key;

    public void AdjustFitness()
    {
        var minFitness = _memberFitness.MinBy(val => val.Value).Value;
        var maxFitness = _memberFitness.MaxBy(val => val.Value).Value;
        var averageFitness = _memberFitness.Values.Average();
        var range = Math.Max(1.0, maxFitness - minFitness);

        Fitness = (averageFitness - minFitness) / range;

        _stagnationManager.Update(maxFitness);
    }

    public NicheReport GetReport() => new()
    {
        Innovation = _innovationId,
        SpeciesId = _speciesId,
        MascotId = Mascot.GenomeId,
        Age = _age,
        AdjustedFitness = Fitness,
        MaxFitness = _memberFitness.Values.Max(),
        MinFitness = _memberFitness.Values.Min(),
        Stagnation = _stagnationManager.Stagnation,
        NumMembers = _memberFitness.Count
    };
}
