using Radiate.Tensors.Enums;

namespace Radiate.Records;

public record TensorTrainOptions(
    int BatchSize = 1,
    int Padding = 0,
    Shape FeatureShape = null,
    float SplitPct = 0f,
    int Layer = 0,
    bool Shuffle = false,
    Norm FeatureNorm = Norm.None,
    Norm TargetNorm = Norm.None,
    SpaceKernel SpaceKernel = null,
    NormalizeScalars FeatureScalars = null,
    NormalizeScalars TargetScalars = null);