using Radiate.Domain.Tensors;

namespace Radiate.Domain.Records;

public record Slice(Tensor Volume, int HStride, int WStride, int DStride = 0);