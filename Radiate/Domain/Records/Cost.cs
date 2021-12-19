
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Records;

public record Cost(Tensor Errors, float loss);
