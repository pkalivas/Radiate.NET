
namespace Radiate.Domain.Records;

public record Epoch(int Index, float AverageLoss = 0f, float ClassificationAccuracy = 0f, float RegressionAccuracy = 0f, float Fitness = 0f);
