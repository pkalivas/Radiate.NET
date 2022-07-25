
using Radiate.Tensors;

namespace Radiate.Records;

public record Prediction(Tensor Result, int Classification = 0, float Confidence = 0f);
