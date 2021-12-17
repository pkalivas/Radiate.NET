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

    public Layer Load()
    {
        if (LayerType == LayerType.Dense)
        {
            return new Dense(Dense.Activation, Dense.Shape, Dense.Weights, Dense.Bias,
                Dense.WeightGradients, Dense.BiasGradients);
        }

        if (LayerType == LayerType.Dropout)
        {
            return new Dropout(Dropout.DropoutRate);
        }

        if (LayerType == LayerType.Flatten)
        {
            return new Flatten(Flatten.Shape, Flatten.PreviousShape);
        }

        if (LayerType == LayerType.LSTM)
        {
            var layer = Lstm;

            var iGate = new Dense(Domain.Activation.Activation.Sigmoid, layer.InputGate.Shape, layer.InputGate.Weights,
                layer.InputGate.Bias, layer.InputGate.WeightGradients, layer.InputGate.BiasGradients);
            var fGate = new Dense(Domain.Activation.Activation.Sigmoid, layer.ForgetGate.Shape, layer.ForgetGate.Weights,
                layer.ForgetGate.Bias, layer.ForgetGate.WeightGradients, layer.ForgetGate.BiasGradients);
            var oGate = new Dense(Domain.Activation.Activation.Sigmoid, layer.OutputGate.Shape, layer.OutputGate.Weights,
                layer.OutputGate.Bias, layer.OutputGate.WeightGradients, layer.OutputGate.BiasGradients);
            var gGate = new Dense(Domain.Activation.Activation.Tanh, layer.GateGate.Shape, layer.GateGate.Weights,
                layer.GateGate.Bias, layer.GateGate.WeightGradients, layer.GateGate.BiasGradients);

            return new LSTM(layer.Shape, layer.CellActivation, layer.HiddenActivation, iGate, fGate, gGate, oGate,
                layer.ForwardTrack, layer.BackwardTrack);
        }

        if (LayerType == LayerType.Conv)
        {
            return new Conv(Conv.Activation, Conv.Shape, Conv.Kernel, Conv.Stride, Conv.Filters,
                Conv.FilterGradients, Conv.Bias, Conv.BiasGradients);
        }

        if (LayerType == LayerType.MaxPool)
        {
            return new MaxPool(MaxPool.Shape, MaxPool.Kernel, MaxPool.Stride);
        }

        throw new Exception($"Cannot load layer {LayerType}");
    }
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
    public Shape PreviousShape { get; set; }
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