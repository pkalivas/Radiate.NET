using Radiate.Domain.Tensors;

namespace Radiate.Domain.Records;

public record Batch(Tensor[] Features, Tensor[] Targets)
{
    public (Shape featureShape, Shape targetShape) InnerShapes => (Features.First().Shape, Targets.First().Shape);

    public int Size => Features.Length;
};