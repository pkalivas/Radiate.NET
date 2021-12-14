using System.Collections.Generic;
using Radiate.NET.Domain.Tensors;

namespace Radiate.NET.Domain.Records
{
    public record Batch(List<Tensor> Features, List<Tensor> Targets);
}