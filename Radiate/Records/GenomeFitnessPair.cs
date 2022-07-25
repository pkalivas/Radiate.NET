
using Radiate.Optimizers.Evolution.Interfaces;

namespace Radiate.Records;

public record GenomeFitnessPair(IGenome Genome)
{
    public float Fitness { get; set; }
}
