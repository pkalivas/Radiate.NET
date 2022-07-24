
using System.Collections.Concurrent;
using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public class Niche : Allele
{
    private readonly StagnationManager _stagnationManager;
    private readonly Guid _nicheId;

    private int _age;

    public Guid Mascot;
    public double TotalAdjustedFitness;
    public readonly ConcurrentBag<NicheMember> Members;
    public readonly ConcurrentBag<NicheMember> AdjustedMembers;

    public Niche(Guid nicheId, NicheMember mascot, StagnationControl stagnationControl)
    {
        _age = 0;
        _stagnationManager = new StagnationManager(stagnationControl);

        Mascot = mascot.MemberId;
        TotalAdjustedFitness = 0;
        _nicheId = nicheId;
        Members = new ConcurrentBag<NicheMember>(new[] { mascot });
        AdjustedMembers = new ConcurrentBag<NicheMember>();
    }

    public bool IsStagnant => _stagnationManager.IsStagnant;
    
    public void AddMember(NicheMember member)
    {
        Members.Add(member);
    }

    public void Reset()
    {
        var randomIdx = Random.Next(Members.Count);
        var newMascot = Members.ElementAt(randomIdx);

        Mascot = newMascot.MemberId;
        _age++;
        TotalAdjustedFitness = 0.0;
        Members.Clear();
        AdjustedMembers.Clear();
    }

    public Guid BestMember() => Members.MaxBy(mem => mem.Fitness)!.MemberId;

    public void CalcTotalAdjustedFitness()
    {
        var tempTotal = 0.0;
        foreach (var member in Members)
        {
            var adjustedFitness = member.Fitness == 0
                ? member.Fitness
                : member.Fitness / Members.Count;

            AdjustedMembers.Add(member with { Fitness = adjustedFitness });
            tempTotal += adjustedFitness;
        }

        TotalAdjustedFitness = tempTotal;
        _stagnationManager.Update(TotalAdjustedFitness);
    }

    public NicheReport GetReport() => new()
    {
        Innovation = InnovationId,
        Id = _nicheId,
        Mascot = Mascot,
        Age = _age,
        AdjustedFitness = TotalAdjustedFitness,
        NumMembers = Members.Count
    };
}
