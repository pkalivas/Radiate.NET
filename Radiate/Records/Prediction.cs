
using Radiate.Tensors;

namespace Radiate.Records;

public record Prediction(Tensor Result, int Classification, float Confidence = 0f);
