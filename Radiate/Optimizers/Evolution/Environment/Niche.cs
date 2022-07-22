
namespace Radiate.Optimizers.Evolution.Environment;

public class Niche
{
    public Guid Mascot { get; set; }
    public List<(Guid memberId, double fitness)> Members { get; set; }
    public int Age { get; set; }
    public double TotalAdjustedFitness { get; set; }
    public Guid NicheId { get; set; }

    public Niche() { }

    public Niche(Guid mascot, double mascotFitness)
    {
        Mascot = mascot;
        Members = new List<(Guid memberId, double fitness)> {(mascot, mascotFitness)};
        Age = 0;
        TotalAdjustedFitness = 0;
        NicheId = Guid.NewGuid();
    }

    public Niche Reset()
    {
        var randomIdx = new Random().Next(Members.Count);
        var newMascot = Members[randomIdx];

        return new Niche
        {
            Mascot = newMascot.memberId,
            Age = ++Age,
            TotalAdjustedFitness = 0.0,
            NicheId = NicheId,
            Members = new List<(Guid memberId, double fitness)>()
        };
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
}
