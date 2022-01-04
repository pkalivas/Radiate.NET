namespace Radiate.Domain.Records;

public record TensorTrainOptions(int BatchSize = 0, int Padding = 0, Shape FeatureShape = null, float SplitPct = 0f, int Layer = 0);