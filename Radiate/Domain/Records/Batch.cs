using Radiate.Domain.Tensors;

namespace Radiate.Domain.Records;

public record Batch(Tensor[] Features, Tensor[] Targets);