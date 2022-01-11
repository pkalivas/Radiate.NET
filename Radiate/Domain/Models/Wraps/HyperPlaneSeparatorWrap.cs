using Radiate.Domain.Gradients;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Supervised.SVM.Info;

namespace Radiate.Domain.Models.Wraps;

public class HyperPlaneWrap
{
    public int FeatureIndex { get; set; }
    public GradientInfo GradientInfo { get; set; }
    public SVMInfo SVMInfo { get; set; }
    public Tensor Weights { get; set; }
}