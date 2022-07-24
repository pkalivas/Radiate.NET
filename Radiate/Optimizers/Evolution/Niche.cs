
using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public class Niche : Allele
{
    private readonly StagnationManager _stagnationManager;

    private int _age;

    public Guid Mascot;
    public Guid NicheId;
    public double TotalAdjustedFitness;
    
    public List<NicheMember> Members { get; }

    public Niche(NicheMember mascot, StagnationControl stagnationControl)
    {
        Mascot = mascot.MemberId;
        Members = new List<NicheMember> { mascot };
        _age = 0;
        TotalAdjustedFitness = 0;
        NicheId = Guid.NewGuid();
        _stagnationManager = new StagnationManager(stagnationControl);
    }

    public bool IsStagnant => _stagnationManager.IsStagnant;
    
    public void AddMember(NicheMember member)
    {
        Members.Add(member);
    }

    public void Reset()
    {
        var randomIdx = Random.Next(Members.Count);
        var newMascot = Members[randomIdx];

        Mascot = newMascot.MemberId;
        _age = _age + 1;
        TotalAdjustedFitness = 0.0;
        Members.Clear();
    }

    public Guid BestMember() => Members.MaxBy(mem => mem.Fitness).MemberId;

    public void CalcTotalAdjustedFitness()
    {
        var tempTotal = 0.0;
        for (var i = 0; i < Members.Count; i++)
        {
            var currentMember = Members[i];

            currentMember = currentMember with
            {
                Fitness = currentMember.Fitness == 0
                    ? currentMember.Fitness
                    : currentMember.Fitness / Members.Count
            };

            tempTotal += currentMember.Fitness;
        }

        TotalAdjustedFitness = tempTotal;
        _stagnationManager.Update(TotalAdjustedFitness);
    }

    public NicheReport GetReport() => new()
    {
        Innovation = InnovationId,
        Id = NicheId,
        Mascot = Mascot,
        Age = _age,
        AdjustedFitness = TotalAdjustedFitness,
        NumMembers = Members.Count
    };
}
