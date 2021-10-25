using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.NET.Engine;
using Radiate.NET.Models.Neat.Enums;

namespace Radiate.NET.Models.Neat.Layers
{
    public class RNNState
    {
        public Stack<List<float>> WGateOutput { get; set; } = new();
    }
    
    public class RNN : Genome, ILayer
    {
        private int InputSize { get; set; }
        private int OutputSize { get; set; }
        private int MemorySize { get; set; }
        private ActivationFunction ActivationFunction { get; set; }
        private List<float> Memory { get; set; }
        private RNNState State { get; set; }
        private Dense UGate { get; set; }
        private Dense WGate { get; set; }
        private Dense VGate { get; set; }

        private RNN() { }
        
        public RNN(int inputSize, int outputSize, int memorySize, ActivationFunction activationFunction)
        {
            InputSize = inputSize;
            OutputSize = outputSize;
            MemorySize = memorySize;
            ActivationFunction = activationFunction;
            Memory = Enumerable.Range(0, memorySize).Select(_ => 0f).ToList();
            State = new RNNState();
            UGate = new Dense(inputSize, memorySize, ActivationFunction.ReLU6);
            WGate = new Dense(memorySize, memorySize, ActivationFunction.Tanh);
            VGate = new Dense(memorySize, outputSize, activationFunction);
        }

        #region Layer Implementation

        public List<float> Forward(List<float> data)
        {
            var uOut = UGate.Forward(data);

            var preActivated = VectorOperations.ElementAdd(uOut, Memory);
            Memory = WGate.Forward(preActivated);

            var vOut = VGate.Forward(Memory);
            
            State.WGateOutput.Push(Memory);

            return vOut;
        }

        public List<float> Backward(List<float> errors, float learningRate)
        {
            var previousMemory = State.WGateOutput.Pop();

            var vOut = VGate.Backward(errors, learningRate);
            var wOut = WGate.Backward(VectorOperations.ElementMultiply(vOut, previousMemory), learningRate);
            var uOut = UGate.Backward(wOut, learningRate);
            
            return uOut;
        }

        public void AddTracer()
        {
            UGate.AddTracer();
            WGate.AddTracer();
            VGate.AddTracer();
        }

        public void RemoveTracer()
        {
            UGate.RemoveTracer();
            WGate.RemoveTracer();
            VGate.RemoveTracer();
        }

        public void Reset()
        {
            Memory = Enumerable.Range(0, MemorySize).Select(_ => 0f).ToList();
            State = new RNNState();
            UGate.Reset();
            WGate.Reset();
            VGate.Reset();
        }

        public LayerType GetLayerType() => LayerType.RNN;

        public ILayer CloneLayer() => new RNN
        {
            InputSize = InputSize,
            OutputSize = OutputSize,
            MemorySize = MemorySize,
            Memory = Enumerable.Range(0, MemorySize).Select(_ => 0f).ToList(),
            State = new RNNState(),
            UGate = UGate.CloneLayer() as Dense,
            WGate = WGate.CloneLayer() as Dense,
            VGate = VGate.CloneLayer() as Dense
        };

        #endregion

        #region Genome Implementation

        public override async Task<T> Crossover<T, TE>(T other, TE environment, double crossoverRate) => new RNN
        {
            InputSize = InputSize,
            OutputSize = OutputSize,
            MemorySize = MemorySize,
            ActivationFunction = ActivationFunction,
            Memory = Enumerable.Range(0, MemorySize).Select(_ => 0f).ToList(),
            State = new RNNState(),
            UGate = await UGate.Crossover((other as RNN).UGate, environment, crossoverRate),
            WGate = await WGate.Crossover((other as RNN).WGate, environment, crossoverRate),
            VGate = await VGate.Crossover((other as RNN).VGate, environment, crossoverRate)
        } as T;

        public override async Task<double> Distance<T, TE>(T other, TE environment)
        {
            var castedOther = other as RNN;

            var tasks = new List<Task<double>>
            {
                UGate.Distance(castedOther.UGate, environment),
                WGate.Distance(castedOther.WGate, environment),
                VGate.Distance(castedOther.VGate, environment)
            };

            return (await Task.WhenAll(tasks)).Sum();
        }

        public override T CloneGenome<T>() => new RNN
        {
            InputSize = InputSize,
            OutputSize = OutputSize,
            MemorySize = MemorySize,
            Memory = Enumerable.Range(0, MemorySize).Select(_ => 0f).ToList(),
            State = new RNNState(),
            UGate = UGate.CloneLayer() as Dense,
            WGate = WGate.CloneLayer() as Dense,
            VGate = VGate.CloneLayer() as Dense
        } as T;

        public override void ResetGenome()
        {
            Reset();
        }

        #endregion
        
    }
}