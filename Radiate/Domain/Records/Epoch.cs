
namespace Radiate.Domain.Records;

public record Epoch(int Index, float AverageLoss, float ClassificationAccuracy, float RegressionAccuracy);
