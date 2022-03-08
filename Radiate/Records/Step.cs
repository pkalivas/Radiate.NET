using Radiate.Tensors;

namespace Radiate.Records;

public record Step(Prediction Prediction, Tensor Target, TimeSpan Time);