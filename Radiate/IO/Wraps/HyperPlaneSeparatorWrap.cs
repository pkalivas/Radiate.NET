using Radiate.Gradients;
using Radiate.Optimizers.Supervised.SVM.Info;
using Radiate.Tensors;

namespace Radiate.IO.Wraps;

public class HyperPlaneWrap
{
    public int FeatureIndex { get; init; }
    public GradientInfo GradientInfo { get; init; }
    public SVMInfo SVMInfo { get; init; }
    public Tensor Weights { get; init; }
}