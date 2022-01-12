
namespace Radiate.Records;

public record Epoch(int Index, 
    float Loss = 0f, 
    float CategoricalAccuracy = 0f, 
    float RegressionAccuracy = 0f,
    float ClassificationAccuracy = 0f,
    float Fitness = 0f);
