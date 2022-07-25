using Radiate.Optimizers.Evolution.Interfaces;

namespace Radiate.Records;

public record Gene(Guid GenomeId, float Fitness, IGenome Genome);