namespace Radiate.Optimizers.Supervised.Perceptrons.Info;

public class SoftMaxInfo : LayerInfo
{
    public int LayerSize { get; set; }

    public SoftMaxInfo(int layerSize = 0)
    {
        LayerSize = layerSize;
    }
}
