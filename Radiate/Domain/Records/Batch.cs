using Radiate.Domain.Tensors;

namespace Radiate.Domain.Records;

public record Batch<T>(List<T> Features = null, List<T> Targets = null)
{
    public List<(TE, TE)> ReadPairs<TE>()
        where TE : class => Features.Zip(Targets)
            .Select(pair => (pair.First as TE, pair.Second as TE))
            .ToList();
}