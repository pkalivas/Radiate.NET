using Radiate.Tensors;

namespace Radiate.Records;

public record Slice(Tensor Volume, int HStride, int WStride, int DStride = 0);