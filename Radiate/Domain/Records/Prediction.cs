
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Records;

public record Prediction(Tensor Result, int Classification, float Confidence = 0f);
