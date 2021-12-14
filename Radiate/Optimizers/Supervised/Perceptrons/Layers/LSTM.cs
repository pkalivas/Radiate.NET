using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Domain.Activation;
using Radiate.Domain.Gradients;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Optimizers.Supervised.Perceptrons.Layers
{
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


        public LSTM(Shape shape, IActivationFunction cellActivation, IActivationFunction hiddenActivation) : base(shape)
        {
            var gateInputSize = shape.Height + shape.Width;

            var gateShape = new Shape(gateInputSize, shape.Width, 0);
            _cellActivation = cellActivation;
            _hiddenActivation = hiddenActivation;
            _inputGate = new Dense(gateShape, new Sigmoid());
            _forgetGate = new Dense(gateShape, new Sigmoid());
            _outputGate = new Dense(gateShape, new Sigmoid());
            _gateGate = new Dense(gateShape, new Tanh());
            _forwardTrack = new Stack<LSTMCell>(new[] { new LSTMCell(shape.Width) });
            _backwardTrack = new Stack<LSTMCell>(new[] { new LSTMCell(shape.Width) });
        }
        
        public override Tensor Predict(Tensor input)
        {
            var prevCell = _forwardTrack.Pop();

            var cellInput = input.ElementsOneD.Concat(prevCell.Hidden.ElementsOneD)
                .ToArray()
                .ToTensor();

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

        public override Tensor FeedForward(Tensor input)
        {
            var prevCell = _forwardTrack.Peek();

            var cellInput = input.ElementsOneD.Concat(prevCell.Hidden.ElementsOneD)
                .ToArray()
                .ToTensor();

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

            var dInput = _cellActivation.Deactivate(current.InputOut) * dI;
            var dForget = _cellActivation.Deactivate(current.ForgetOut) * dF;
            var dOutput = _cellActivation.Deactivate(current.OutputOut) * dO;
            var dGate = _hiddenActivation.Deactivate(current.GateOut) * dG;

            var iE = _inputGate.PassBackward(dInput);
            var fE = _forgetGate.PassBackward(dForget);
            var oE = _outputGate.PassBackward(dOutput);
            var gE = _gateGate.PassBackward(dGate);

            var dx = (iE + fE + oE + gE).ElementsOneD;

            var cellGrad = dS * current.ForgetOut;
            var hiddenGrad = dx.Skip(Shape.Height).Take(Shape.Width).ToArray();

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
}