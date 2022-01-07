namespace Radiate.Domain.Records;

public record Validation(float Loss, float ClassificationAccuracy, float RegressionAccuracy, float CategoricalAccuracy);