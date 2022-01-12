using Radiate.Tensors;

namespace Radiate.Records;

public record Batch(Tensor[] Features, Tensor[] Targets = null);