using Radiate.NET.Domain.Activation;

namespace Radiate.NET.Optimizers.Perceptrons.Info
{
    public class DenseInfo : LayerInfo
    {
        public int LayerSize { get; set; }
        public Activation Activation { get; set; }

        public DenseInfo(Activation activation)
        {
            Activation = activation;
        }
        
        public DenseInfo(int layerSize, Activation activation)
        {
            LayerSize = layerSize;
            Activation = activation;
        }
    };
}