using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons.Layers;

public class LSTM : Layer
{
    private readonly IActivationFunction _cellActivation;
    private readonly IActivationFunction _hiddenActivation;
    private readonly Shape _memoryShape;
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
        _memoryShape = wrap.MemoryShape;
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
        _memoryShape = gateShape;
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

    public override async Task<Tensor> PassBackward(Tensor errors)
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

        var dInput = _cellActivation.Deactivate(current.InputOut) * dI;
        var dForget = _cellActivation.Deactivate(current.ForgetOut) * dF;
        var dOutput = _cellActivation.Deactivate(current.OutputOut) * dO;
        var dGate = _hiddenActivation.Deactivate(current.GateOut) * dG;

        var backPasses = new List<Task<Tensor>>()
        {
            _inputGate.PassBackward(dInput),
            _forgetGate.PassBackward(dForget),
            _outputGate.PassBackward(dOutput),
            _gateGate.PassBackward(dGate)
        };

        var dx = (await Task.WhenAll(backPasses))
            .Aggregate(Tensor.Like(_memoryShape), (all, curr) => all + curr).Read1D();

        var cellGrad = dS * current.ForgetOut;
        var hiddenGrad = dx.Skip(Shape.Height).Take(Shape.Width);

        current.CellGradient = cellGrad;
        current.HiddenGradient = hiddenGrad.ToTensor();
        
        _backwardTrack.Push(current);

        return current.HiddenGradient;
    }

    public override Task UpdateWeights(GradientInfo info, int epoch)
    {
        _inputGate.UpdateWeights(info, epoch);
        _forgetGate.UpdateWeights(info, epoch);
        _gateGate.UpdateWeights(info, epoch);
        _outputGate.UpdateWeights(info, epoch);
        
        _forwardTrack.Clear();
        _backwardTrack.Clear();
        
        _forwardTrack.Push(new LSTMCell(Shape.Width));
        _backwardTrack.Push(new LSTMCell(Shape.Width));

        return Task.CompletedTask;
    }
    
    public override LayerWrap Save() => new()
    {
        LayerType = LayerType.LSTM,
        Lstm = new LSTMWrap
        {
            Shape = Shape,
            CellActivation = _cellActivation.GetType(),
            HiddenActivation = _hiddenActivation.GetType(),
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
        var cellInput = input.Read1D().Concat(prevCell.Hidden.Read1D()).ToTensor();

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
    public Tensor InputOut { get; set; }
    public Tensor ForgetOut { get; set; }
    public Tensor GateOut { get; set; }
    public Tensor OutputOut { get; set; }
    public Tensor LayerInput { get; set; }
    public Tensor Hidden { get; set; }
    public Tensor Cell { get; set; }
    public Tensor PreviousHidden { get; set; }
    public Tensor PreviousCell { get; set; }
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
