using Radiate.Domain.Tensors;

namespace Radiate.Domain.Records;

public record TrainTestSplit(List<Tensor> Features, List<Tensor> Targets);