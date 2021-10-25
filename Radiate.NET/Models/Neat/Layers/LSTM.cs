using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.NET.Engine;
using Radiate.NET.Models.Neat.Enums;

namespace Radiate.NET.Models.Neat.Layers
{
    public class LSTMState
    {
        public Stack<List<float>> FGateOutput { get; set; } = new();
        public Stack<List<float>> IGateOutput { get; set; } = new();
        public Stack<List<float>> SGateOutput { get; set; } = new();
        public Stack<List<float>> OGateOutput { get; set; } = new();
        public Stack<List<float>> MemoryState { get; set; } = new();
        public List<float> PreviousMemoryDerivative { get; set; }
        public List<float> PreviousHiddenDerivative { get; set; }

        public LSTMState(int memorySize)
        {
            PreviousHiddenDerivative = Enumerable.Range(0, memorySize).Select(_ => 0f).ToList();
            PreviousMemoryDerivative = Enumerable.Range(0, memorySize).Select(_ => 0f).ToList();
        }
    }
    
    public class LSTM : Genome, ILayer
    {
        private int InputSize { get; set; }
        private int MemorySize { get; set; }
        private int OutputSize { get; set; }
        private ActivationFunction ActivationFunction { get; set; }
        private List<float> Memory { get; set; }
        private List<float> Hidden { get; set; }
        private LSTMState States { get; set; }
        private Dense GGate { get; set; }
        private Dense IGate { get; set; }
        private Dense FGate { get; set; }
        private Dense OGate { get; set; }
        private Dense VGate { get; set; }
        
        private LSTM() { }

        public LSTM(int inputSize, int outputSize, int memorySize, ActivationFunction activationFunction)
        {
            var cellInputSize = inputSize + memorySize;

            InputSize = inputSize;
            OutputSize = outputSize;
            MemorySize = memorySize;
            ActivationFunction = activationFunction;
            States = new LSTMState(MemorySize);
            Memory = Enumerable.Range(0, memorySize).Select(idx => 0f).ToList();
            Hidden = Enumerable.Range(0, memorySize).Select(idx => 0f).ToList();
            GGate = new Dense(cellInputSize, memorySize, ActivationFunction.Tanh);
            IGate = new Dense(cellInputSize, memorySize, ActivationFunction.Sigmoid);
            FGate = new Dense(cellInputSize, memorySize, ActivationFunction.Sigmoid);
            OGate = new Dense(cellInputSize, memorySize, ActivationFunction.Sigmoid);
            VGate = new Dense(memorySize, outputSize, activationFunction);
        }

        private bool HasTracers() =>
            GGate.HasTracer() &&
            IGate.HasTracer() &&
            FGate.HasTracer() &&
            OGate.HasTracer() &&
            VGate.HasTracer();


        #region Layer Implementation
        
        public List<float> Forward(List<float> data)
        {
            var inputValues = Hidden.Concat(data).ToList();

            var gGateTask = Task.Run(() => GGate.Forward(inputValues));
            var iGateTask = Task.Run(() => IGate.Forward(inputValues));
            var fGateTask = Task.Run(() => FGate.Forward(inputValues));
            var oGateTask = Task.Run(() => OGate.Forward(inputValues));

            var currentState = gGateTask.ConfigureAwait(true).GetAwaiter().GetResult();
            var currentIGate = iGateTask.ConfigureAwait(true).GetAwaiter().GetResult();
            var currentFGate = fGateTask.ConfigureAwait(true).GetAwaiter().GetResult();
            var currentOutput = oGateTask.ConfigureAwait(true).GetAwaiter().GetResult();

            Memory = VectorOperations.ElementMultiply(Memory, currentFGate);
            var newState = VectorOperations.ElementMultiply(currentState, currentIGate);
            Memory = VectorOperations.ElementAdd(Memory, newState);
            
            var newMemory = VectorOperations.ElementActivate(Memory, ActivationFunction.Tanh);
            currentOutput = VectorOperations.ElementMultiply(currentOutput, newMemory);

            // If the gates have tracers, the model is being trained so we need to keep
            // track of the previous states of the gates
            if (HasTracers())
            {
                States.FGateOutput.Push(currentFGate);
                States.IGateOutput.Push(currentIGate);
                States.SGateOutput.Push(currentState);
                States.OGateOutput.Push(currentOutput);
                States.MemoryState.Push(Memory);   
            }

            Hidden = currentOutput;

            return VGate.Forward(currentOutput);
        }

        public List<float> Backward(List<float> errors, float learningRate)
        {
            // get the derivative of the cell and hidden state from the previous step as well as the previous memory state
            var dhNext = States.PreviousHiddenDerivative;
            var dcNext = States.PreviousMemoryDerivative;

            // unpack the current gate outputs 
            var cOld = States.MemoryState.Pop();
            var gCurr = States.SGateOutput.Pop();
            var iCurr = States.IGateOutput.Pop();
            var fCurr = States.FGateOutput.Pop();
            var oCurr = States.OGateOutput.Pop();
            
            // compute the hidden to output gradient
            // dh = error @ Wy.T + dh_next
            var dh = VectorOperations.ElementAdd(VGate.Backward(errors, learningRate), dhNext);
            
            // Gradient for ho in h = ho * tanh(c)     
            //dho = tanh(c) * dh
            //dho = dsigmoid(ho) * dho
            var dho = VectorOperations.ElementActivate(cOld, ActivationFunction.Tanh);
            dho = VectorOperations.ElementMultiply(dho, dh);
            dho = VectorOperations.ElementMultiply(dho, VectorOperations.ElementDeactivate(oCurr, ActivationFunction.Sigmoid));
            var oTask = Task.Run(() => OGate.Backward(dho, learningRate));
            
            // Gradient for c in h = ho * tanh(c), note we're adding dc_next here     
            // dc = ho * dh * dtanh(c)
            // dc = dc + dc_next
            var dc = VectorOperations.ElementMultiply(oCurr, dh);
            dc = VectorOperations.ElementMultiply(dc, VectorOperations.ElementDeactivate(cOld, ActivationFunction.Tanh));
            dc = VectorOperations.ElementAdd(dc, dcNext);
            
            // Gradient for hf in c = hf * c_old + hi * hc    
            // dhf = c_old * dc
            // dhf = dsigmoid(hf) * dhf
            var dhf = VectorOperations.ElementMultiply(cOld, dc);
            dhf = VectorOperations.ElementMultiply(dhf, VectorOperations.ElementDeactivate(fCurr, ActivationFunction.Sigmoid));
            var fTask = Task.Run(() => FGate.Backward(dhf, learningRate));
            
            // Gradient for hi in c = hf * c_old + hi * hc     
            // dhi = hc * dc
            // dhi = dsigmoid(hi) * dhi
            var dhi = VectorOperations.ElementMultiply(gCurr, dc);
            dhi = VectorOperations.ElementMultiply(dhi, VectorOperations.ElementDeactivate(iCurr, ActivationFunction.Sigmoid));
            var iTask = Task.Run(() => IGate.Backward(dhi, learningRate));
            
            // Gradient for hc in c = hf * c_old + hi * hc     
            // dhc = hi * dc
            // dhc = dtanh(hc) * dhc
            var dhc = VectorOperations.ElementMultiply(iCurr, dc);
            dhc = VectorOperations.ElementMultiply(dhc, VectorOperations.ElementDeactivate(gCurr, ActivationFunction.Sigmoid));
            var gTask = Task.Run(() => GGate.Backward(dhc, learningRate));
            
            // As X was used in multiple gates, the gradient must be accumulated here     
            // dX = dXo + dXc + dXi + dXf
            var dx = Enumerable.Range(0, InputSize + MemorySize).Select(_ => 0f).ToList();
            dx = VectorOperations.ElementAdd(dx, oTask.ConfigureAwait(true).GetAwaiter().GetResult());
            dx = VectorOperations.ElementAdd(dx, fTask.ConfigureAwait(true).GetAwaiter().GetResult());
            dx = VectorOperations.ElementAdd(dx, iTask.ConfigureAwait(true).GetAwaiter().GetResult());
            dx = VectorOperations.ElementAdd(dx, gTask.ConfigureAwait(true).GetAwaiter().GetResult());
            
            // Split the concatenated X, so that we get our gradient of h_old     
            // dh_next = dx[:, :H]
            dhNext = dx.Skip(InputSize).ToList();
            dcNext = VectorOperations.ElementMultiply(fCurr, dc);

            // Gradient for c_old in c = hf * c_old + hi * hc     
            // dc_next = hf * dc
            States.PreviousHiddenDerivative = dhNext;
            States.PreviousMemoryDerivative = dcNext;
            
            // return the error of the input given to the layer
            return dx.Skip(MemorySize).ToList();
        }
        
        public void AddTracer()
        {
            GGate.AddTracer();
            IGate.AddTracer();
            FGate.AddTracer();
            OGate.AddTracer();
            VGate.AddTracer();
        }

        public void RemoveTracer()
        {
            GGate.RemoveTracer();
            IGate.RemoveTracer();
            FGate.RemoveTracer();
            OGate.RemoveTracer();
            VGate.RemoveTracer();
        }

        public void Reset()
        {
            GGate.Reset();
            IGate.Reset();
            FGate.Reset();
            OGate.Reset();
            VGate.Reset();
            States = new LSTMState(MemorySize);
            Memory = Enumerable.Range(0, MemorySize).Select(idx => 0f).ToList();
            Hidden = Enumerable.Range(0, MemorySize).Select(idx => 0f).ToList();
        }

        public LayerType GetLayerType() => LayerType.LSTM;

        public ILayer CloneLayer() => new LSTM
        {
            InputSize = InputSize,
            OutputSize = OutputSize,
            MemorySize = MemorySize,
            ActivationFunction = ActivationFunction,
            Memory = Enumerable.Range(0, MemorySize).Select(idx => 0f).ToList(),
            Hidden = Enumerable.Range(0, MemorySize).Select(idx => 0f).ToList(),
            States = new LSTMState(MemorySize),
            GGate = GGate.CloneLayer() as Dense,
            IGate = IGate.CloneLayer() as Dense,
            FGate = FGate.CloneLayer() as Dense,
            OGate = OGate.CloneLayer() as Dense,
            VGate = VGate.CloneLayer() as Dense
        };

        #endregion

        #region Genome Implementation

        public override async Task<T> Crossover<T, TE>(T other, TE environment, double crossoverRate) => new LSTM
        {
            InputSize = InputSize,
            OutputSize = OutputSize,
            MemorySize = MemorySize,
            ActivationFunction = ActivationFunction,
            Memory = Enumerable.Range(0, MemorySize).Select(idx => 0f).ToList(),
            Hidden = Enumerable.Range(0, MemorySize).Select(idx => 0f).ToList(),
            States = new LSTMState(MemorySize),
            GGate = await GGate.Crossover((other as LSTM).GGate, environment, crossoverRate),
            IGate = await IGate.Crossover((other as LSTM).IGate, environment, crossoverRate),
            FGate = await FGate.Crossover((other as LSTM).FGate, environment, crossoverRate),
            OGate = await OGate.Crossover((other as LSTM).OGate, environment, crossoverRate),
            VGate = await VGate.Crossover((other as LSTM).VGate, environment, crossoverRate)
        } as T;

        public override async Task<double> Distance<T, TE>(T other, TE environment)
        {
            var tasks = new List<Task<double>>()
            {
                GGate.Distance((other as LSTM).GGate, environment),
                IGate.Distance((other as LSTM).IGate, environment),
                FGate.Distance((other as LSTM).FGate, environment),
                OGate.Distance((other as LSTM).OGate, environment),
                VGate.Distance((other as LSTM).VGate, environment)
            };

            return (await Task.WhenAll(tasks)).Sum();
        }

        public override T CloneGenome<T>() => new LSTM
        {
            InputSize = InputSize,
            OutputSize = OutputSize,
            MemorySize = MemorySize,
            ActivationFunction = ActivationFunction,
            Memory = Enumerable.Range(0, MemorySize).Select(idx => 0f).ToList(),
            Hidden = Enumerable.Range(0, MemorySize).Select(idx => 0f).ToList(),
            States = new LSTMState(MemorySize),
            GGate = GGate.CloneLayer() as Dense,
            IGate = IGate.CloneLayer() as Dense,
            FGate = FGate.CloneLayer() as Dense,
            OGate = OGate.CloneLayer() as Dense,
            VGate = VGate.CloneLayer() as Dense
        } as T;

        public override void ResetGenome()
        {
            Reset();
        }

        #endregion
        
    }
}