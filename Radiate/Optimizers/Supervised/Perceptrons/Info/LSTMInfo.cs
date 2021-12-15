using Radiate.Domain.Activation;

namespace Radiate.Optimizers.Supervised.Perceptrons.Info;

public class LSTMInfo : LayerInfo
{
    public int LayerSize { get; set; }
    public int MemorySize { get; set; }
    public Activation CellActivation { get; set; }
    public Activation HiddenActivation { get; set; }


    public LSTMInfo(
        int layerSize, 
        int memorySize, 
        Activation cellActivation = Activation.Sigmoid, 
        Activation hiddenActivation = Activation.Tanh)
    {
        LayerSize = layerSize;
        MemorySize = memorySize;
        CellActivation = cellActivation;
        HiddenActivation = hiddenActivation;
    }
}
