
namespace Radiate.Optimizers.Evolution;

public class Niche : Allele
{
    public Guid Mascot;
    public List<(Guid memberId, double fitness)> Members;
    public int Age;
    public double TotalAdjustedFitness;
    public Guid NicheId;
    
    public Niche(Guid mascot, double mascotFitness)
    {
        Mascot = mascot;
        Members = new List<(Guid memberId, double fitness)> {(mascot, mascotFitness)};
        Age = 0;
        TotalAdjustedFitness = 0;
        NicheId = Guid.NewGuid();
    }

    public void Reset()
    {
        var randomIdx = Random.Next(Members.Count);
        var newMascot = Members[randomIdx];

        Mascot = newMascot.memberId;
        Age = Age + 1;
        TotalAdjustedFitness = 0.0;
        Members.Clear();
    }

    public Guid BestMember() => Members.MaxBy(mem => mem.fitness).memberId;

    public void CalcTotalAdjustedFitness()
    {
        var tempTotal = 0.0;
        for (var i = 0; i < Members.Count; i++)
        {
            var currentMember = Members[i];

            currentMember.fitness = currentMember.fitness == 0
                ? currentMember.fitness
                : currentMember.fitness / Members.Count;

            tempTotal += currentMember.fitness;
        }

        TotalAdjustedFitness = tempTotal;
    }

    public NicheReport GetReport() => new()
    {
        Innovation = InnovationId,
        Id = NicheId,
        Mascot = Mascot,
        Age = Age,
        AdjustedFitness = TotalAdjustedFitness,
        NumMembers = Members.Count
    };
}
