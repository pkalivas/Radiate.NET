using Radiate.Tensors;

namespace Radiate.Records;

public record TrainTestSplit(List<Tensor> Features, List<Tensor> Targets);