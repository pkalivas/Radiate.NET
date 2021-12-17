using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers.Supervised.Perceptrons.Layers;

namespace Radiate.Domain.Models;

public class LayerWrap
{
    public LayerType LayerType { get; set; }
    public DenseWrap Dense { get; set; }
    public DropoutWrap Dropout { get; set; }
    public FlattenWrap Flatten { get; set; }
    public LSTMWrap Lstm { get; set; }
    public MaxPoolWrap MaxPool { get; set; }
    public ConvWrap Conv { get; set; }
}

public class MaxPoolWrap
{
    public Shape Shape { get; set; }
    public Kernel Kernel { get; set; }
    public int Stride { get; set; }
}

public class LSTMWrap
{
    public Shape Shape { get; set; }
    public Activation.Activation CellActivation { get; set; }
    public Activation.Activation HiddenActivation { get; set; }
    public DenseWrap InputGate { get; set; }
    public DenseWrap ForgetGate { get; set; }
    public DenseWrap GateGate { get; set; }
    public DenseWrap OutputGate { get; set; }
    public Stack<LSTMCell> ForwardTrack { get; set; }
    public Stack<LSTMCell> BackwardTrack { get; set; }
}

public class DropoutWrap
{
    public float DropoutRate { get; set; }
}

public class FlattenWrap
{
    public Shape Shape { get; set; }
}

public class DenseWrap
{
    public Activation.Activation Activation { get; set; }
    public Shape Shape { get; set; }
    public Tensor Weights { get; set; }
    public Tensor Bias { get; set; }
    public Tensor WeightGradients { get; set; }
    public Tensor BiasGradients { get; set; }
}

public class ConvWrap
{
    public Shape Shape { get; set; }
    public Kernel Kernel { get; set; }
    public Activation.Activation Activation { get; set; }
    public int Stride { get; set; }
    public Tensor[] Filters { get; set; }
    public Tensor[] FilterGradients { get; set; }
    public Tensor Bias { get; set; }
    public Tensor BiasGradients { get; set; }
}