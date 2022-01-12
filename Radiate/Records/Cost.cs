
using Radiate.Tensors;

namespace Radiate.Records;

public record Cost(Tensor Errors, float Loss);
