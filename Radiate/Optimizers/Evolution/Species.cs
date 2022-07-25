
using System.Collections.Concurrent;
using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public class Species
{
    private readonly Guid _speciesId;
    private readonly ConcurrentDictionary<Guid, double> _memberFitness;
    private readonly int _age;
    private readonly int _innovationId;
    
    public Species(Guid speciesId, Gene mascot)
    {
        var (memberId, memberFitness, _) = mascot;

        Mascot = mascot;
        _speciesId = speciesId;
        _innovationId = InnovationCounter.Increment();
        _memberFitness = new ConcurrentDictionary<Guid, double>(new[] { new KeyValuePair<Guid, double>(memberId, memberFitness) });
    }

    public Species(Species other, Gene mascot)
    {
        var (mascotId, mascotFitness, _) = mascot;

        Mascot = mascot;
        _speciesId = other._speciesId;
        _age = other._age + 1;
        _innovationId = other._innovationId;
        _memberFitness = new ConcurrentDictionary<Guid, double>(new[] { new KeyValuePair<Guid, double>(mascotId, mascotFitness) });
    }
    
    public double Fitness { get; private set; }
    public Gene Mascot { get; }
    
    public IEnumerable<(Guid GenomeId, double Fitness)> Members => _memberFitness.Select(pair => (pair.Key, pair.Value));
    
    public Guid BestMember() => _memberFitness.MaxBy(mem => mem.Value).Key;

    public void AddMember(SpeciesMember member)
    {
        _memberFitness[member.GenomeId] = member.Fitness;
    }
    
    public double AdjustFitness()
    {
        var minFitness = _memberFitness.MinBy(val => val.Value).Value;
        var maxFitness = _memberFitness.MaxBy(val => val.Value).Value;
        var averageFitness = _memberFitness.Values.Average();
        var range = Math.Max(1.0, maxFitness - minFitness);

        Fitness = (averageFitness - minFitness) / range;
        
        return maxFitness;
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
        NumMembers = _memberFitness.Count
    };
}
