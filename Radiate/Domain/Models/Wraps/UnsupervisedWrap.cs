using Radiate.Domain.Tensors;

namespace Radiate.Domain.Models.Wraps;


public class KMeansWrap
{
    public int KClusters { get; set; }
    public List<int>[] Clusters { get; set; }
    public Tensor[] Centroids { get; set; }
}