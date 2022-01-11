using Radiate.Domain.Activation;
using Radiate.Domain.Extensions;
using Radiate.Domain.Gradients;
using Radiate.Domain.Models;
using Radiate.Domain.Models.Wraps;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons.Layers;

public class LSTM : Layer
{
    private readonly IActivationFunction _cellActivation;
    private readonly IActivationFunction _hiddenActivation;
    private readonly Dense _inputGate;
    private readonly Dense _forgetGate;
    private readonly Dense _gateGate;
    private readonly Dense _outputGate;
    private readonly Stack<LSTMCell> _forwardTrack;
    private readonly Stack<LSTMCell> _backwardTrack;

    public LSTM(LSTMWrap wrap) : base(wrap.Shape)
    {
        _cellActivation = ActivationFunctionFactory.Get(wrap.CellActivation);
        _hiddenActivation = ActivationFunctionFactory.Get(wrap.HiddenActivation);
        _inputGate = new Dense(wrap.InputGate);
        _forgetGate = new Dense(wrap.ForgetGate);
        _gateGate = new Dense(wrap.GateGate);
        _outputGate = new Dense(wrap.OutputGate);
        _forwardTrack = wrap.ForwardTrack;
        _backwardTrack = wrap.BackwardTrack;
    }

    public LSTM(Shape shape, IActivationFunction cellActivation, IActivationFunction hiddenActivation) : base(shape)
    {
        var gateInputSize = shape.Height + shape.Width;

        var gateShape = new Shape(gateInputSize, shape.Width);
        _cellActivation = cellActivation;
        _hiddenActivation = hiddenActivation;
        _inputGate = new Dense(gateShape, new Sigmoid());
        _forgetGate = new Dense(gateShape, new Sigmoid());
        _outputGate = new Dense(gateShape, new Sigmoid());
        _gateGate = new Dense(gateShape, new Tanh());
        _forwardTrack = new Stack<LSTMCell>(new[] { new LSTMCell(shape.Width) });
        _backwardTrack = new Stack<LSTMCell>(new[] { new LSTMCell(shape.Width) });
    }

    public override Tensor Predict(Tensor input) => OperateGates(input, _forwardTrack.Pop());

    public override Tensor FeedForward(Tensor input) => OperateGates(input, _forwardTrack.Peek());

    public override Tensor PassBackward(Tensor errors)
    {
        var current = _forwardTrack.Pop();
        var previous = _backwardTrack.Peek();
        
        var dH = errors + previous.HiddenGradient;
        var dC = _hiddenActivation.Deactivate(previous.CellGradient);
        
        var dS = current.OutputOut * dH * dC;
        var dO = current.Cell * dH;
        var dI = current.GateOut * dS;
        var dG = current.InputOut * dS;
        var dF = current.PreviousCell * dS;

        dI.Multiply(_cellActivation.Deactivate(current.InputOut));
        dF.Multiply(_cellActivation.Deactivate(current.ForgetOut));
        dO.Multiply(_cellActivation.Deactivate(current.OutputOut));
        dG.Multiply(_hiddenActivation.Deactivate(current.GateOut));
        
        var iE = _inputGate.PassBackward(dI);
        var fE = _forgetGate.PassBackward(dF);
        var oE = _outputGate.PassBackward(dO);
        var gE = _gateGate.PassBackward(dG);

        var dx = Tensor.Like(iE.Shape);
        dx.Add(iE);
        dx.Add(fE);
        dx.Add(oE);
        dx.Add(gE);

        var cellGrad = dS * current.ForgetOut;
        var hiddenGrad = dx.Skip(Shape.Height).Take(Shape.Width);

        current.CellGradient = cellGrad;
        current.HiddenGradient = hiddenGrad.ToTensor();
        
        _backwardTrack.Push(current);

        return current.HiddenGradient;
    }

    public override void UpdateWeights(GradientInfo info, int epoch, int batchSize)
    {
        var gates = new List<Dense>()
        {
            _inputGate,
            _forgetGate,
            _gateGate,
            _outputGate
        };

        Parallel.ForEach(gates, gate => gate.UpdateWeights(info, epoch, batchSize));

        _forwardTrack.Clear();
        _backwardTrack.Clear();
        
        _forwardTrack.Push(new LSTMCell(Shape.Width));
        _backwardTrack.Push(new LSTMCell(Shape.Width));
    }
    
    public override LayerWrap Save() => new()
    {
        LayerType = LayerType.LSTM,
        Lstm = new LSTMWrap
        {
            Shape = Shape,
            CellActivation = _cellActivation.ActivationType(),
            HiddenActivation = _hiddenActivation.ActivationType(),
            InputGate = _inputGate.Save().Dense,
            ForgetGate = _forgetGate.Save().Dense,
            GateGate = _gateGate.Save().Dense,
            OutputGate = _outputGate.Save().Dense,
            ForwardTrack = _forwardTrack,
            BackwardTrack = _backwardTrack
        }
    };

    private Tensor OperateGates(Tensor input, LSTMCell prevCell)
    {
        var cellInput = input.Concat(prevCell.Hidden).ToTensor();

        var gOut = _gateGate.FeedForward(cellInput);
        var iOut = _inputGate.FeedForward(cellInput);
        var fOut = _forgetGate.FeedForward(cellInput);
        var oOut = _outputGate.FeedForward(cellInput);

        var currentCellState = fOut * prevCell.Cell + gOut * iOut;
        var currentCellHidden = _hiddenActivation.Activate(currentCellState) + oOut;
        
        _forwardTrack.Push(new LSTMCell
        {
            Cell = currentCellState,
            Hidden = currentCellHidden,
            ForgetOut = fOut,
            GateOut = gOut,
            InputOut = iOut,
            OutputOut = oOut,
            PreviousCell = prevCell.Cell,
            PreviousHidden = prevCell.Hidden
        });

        return currentCellHidden;
    }
}


public class LSTMCell
{
    public Tensor InputOut { get; init; }
    public Tensor ForgetOut { get; init; }
    public Tensor GateOut { get; init; }
    public Tensor OutputOut { get; init; }
    public Tensor Hidden { get; init; }
    public Tensor Cell { get; init; }
    public Tensor PreviousHidden { get; set; }
    public Tensor PreviousCell { get; init; }
    public Tensor HiddenGradient { get; set; }
    public Tensor CellGradient { get; set; }

    public LSTMCell() { }
    
    public LSTMCell(int memorySize)
    {
        Hidden = new float[memorySize].ToTensor();
        Cell = new float[memorySize].ToTensor();
        PreviousCell = new float[memorySize].ToTensor();
        PreviousHidden = new float[memorySize].ToTensor();
        HiddenGradient = new float[memorySize].ToTensor();
        CellGradient = new float[memorySize].ToTensor();
    }
}
