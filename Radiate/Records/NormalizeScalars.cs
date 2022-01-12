namespace Radiate.Records;

public record NormalizeScalars(Dictionary<int, float> MinLookup, Dictionary<int, float> MaxLookup, Dictionary<int, float> MeanLookup, Dictionary<int, float> StdLookup);