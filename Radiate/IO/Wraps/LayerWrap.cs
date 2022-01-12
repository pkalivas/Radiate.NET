using Radiate.Activations;
using Radiate.Optimizers.Supervised.Perceptrons.Layers;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.IO.Wraps;

public class LayerWrap
{
    public LayerType LayerType { get; init; }
    public DenseWrap Dense { get; init; }
    public DropoutWrap Dropout { get; init; }
    public FlattenWrap Flatten { get; init; }
    public LSTMWrap Lstm { get; init; }
    public MaxPoolWrap MaxPool { get; init; }
    public ConvWrap Conv { get; init; }
}

public class MaxPoolWrap
{
    public Shape Shape { get; init; }
    public Kernel Kernel { get; init; }
    public int Stride { get; init; }
}

public class LSTMWrap
{
    public Shape Shape { get; init; }
    public Activation CellActivation { get; init; }
    public Activation HiddenActivation { get; init; }
    public DenseWrap InputGate { get; init; }
    public DenseWrap ForgetGate { get; init; }
    public DenseWrap GateGate { get; init; }
    public DenseWrap OutputGate { get; init; }
    public Stack<LSTMCell> ForwardTrack { get; init; }
    public Stack<LSTMCell> BackwardTrack { get; init; }
}

public class DropoutWrap
{
    public float DropoutRate { get; init; }
}

public class FlattenWrap
{
    public Shape Shape { get; init; }
}

public class DenseWrap
{
    public Activation Activation { get; init; }
    public Shape Shape { get; init; }
    public Tensor Weights { get; init; }
    public Tensor Bias { get; init; }
    public Tensor WeightGradients { get; init; }
    public Tensor BiasGradients { get; init; }
}

public class ConvWrap
{
    public Shape Shape { get; init; }
    public Kernel Kernel { get; init; }
    public Activation Activation { get; init; }
    public int Stride { get; init; }
    public Tensor[] Filters { get; init; }
    public Tensor[] FilterGradients { get; init; }
    public Tensor Bias { get; init; }
    public Tensor BiasGradients { get; init; }
}