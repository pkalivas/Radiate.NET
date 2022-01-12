using Radiate.Tensors;

namespace Radiate.IO.Wraps;


public class KMeansWrap
{
    public int KClusters { get; init; }
    public List<int>[] Clusters { get; init; }
    public Tensor[] Centroids { get; init; }
}