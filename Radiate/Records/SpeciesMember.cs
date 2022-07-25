namespace Radiate.Records;

public record SpeciesMember(Guid GenomeId, double Fitness);

public record SpeciesCandidate(Guid SpeciesId, double Distance);