using Radiate.Domain.Tensors;

namespace Radiate.Domain.Records;

public record Batch(List<Tensor> Features, List<Tensor> Targets);