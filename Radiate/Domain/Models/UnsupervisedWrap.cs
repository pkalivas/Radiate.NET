using Radiate.Domain.Tensors;
using Radiate.Optimizers.Unsupervised;

namespace Radiate.Domain.Models;

public class UnsupervisedWrap
{
    public UnsupervisedType UnsupervisedType { get; set; }
    public KMeansWrap KMeansWrap { get; set; }
}

public class KMeansWrap
{
    public int KClusters { get; set; }
    public List<int>[] Clusters { get; set; }
    public Tensor[] Centroids { get; set; }
}