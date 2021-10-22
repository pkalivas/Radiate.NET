using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.NET.Engine;
using Radiate.NET.Enums;
using Radiate.NET.Models.Neat.Enums;
using Radiate.NET.Models.Neat.Layers;

namespace Radiate.NET.Models.Neat
{
    public class Neat : Genome
    {
        private List<ILayer> Layers { get; set; }
        private int BatchSize { get; set; }

        public Neat()
        {
            Layers = new List<ILayer>();
            BatchSize = 1;
        }


        public Neat AddLayer(ILayer layer)
        {
            Layers.Add(layer);
            return this;
        }

        public Neat SetBatchSize(int size)
        {
            BatchSize = size;
            return this;
        }

        public void Train(List<List<float>> inputs, List<List<float>> targets, float learningRate, Func<int, float, bool> func)
        {
            var passOutput = new List<List<float>>();
            var passTarget = new List<List<float>>();
            var epoch = 0;
            var count = 0;
            var loss = 0f;

            if (BatchSize > 1)
            {
                foreach (var layer in Layers)
                {
                    layer.AddTracer();
                }
            }

            while (true)
            {
                foreach (var (input, idx) in inputs.Select((input, idx) => (input, idx)))
                {
                    count++;

                    passOutput.Add(Forward(input));
                    passTarget.Add(targets[idx]);

                    if (count == BatchSize || idx == inputs.Count - 1)
                    {
                        count = 0;
                        loss += Backward(passOutput, passTarget, learningRate);
                        passOutput = new List<List<float>>();
                        passTarget = new List<List<float>>();
                    }
                }

                if (func(epoch, loss))
                {
                    break;
                }
                
                epoch++;
                loss = 0f;
            }

            foreach (var layer in Layers)
            {
                layer.RemoveTracer();
            }
        }

        public List<float> Forward(List<float> data) => Layers.Aggregate(data, (current, layer) => layer.Forward(current));

        private float Backward(IReadOnlyList<List<float>> modelOutputs, IReadOnlyList<List<float>> networkTargets, float learningRate)
        {
            var totalLoss = 0f;
            for (var i = modelOutputs.Count - 1; i >= 0; i--)
            {
                var (loss, errors) = VectorOperations.GetLoss(networkTargets[i], modelOutputs[i], LossFunction.Difference);
                totalLoss += loss;

                for (var j = Layers.Count - 1; j >= 0; j--)
                {
                    errors = Layers[j].Backward(errors, learningRate);
                }
            }
            
            ResetGenome();

            return totalLoss;
        }
        
        public override async Task<T> Crossover<T, TE>(T other, TE environment, double crossoverRate)
        {
            var parentTwo = other as Neat;
            var neatEnv = environment as NeatEnvironment;

            var childLayers = new List<ILayer>();
            foreach (var layers in Layers.Zip(parentTwo.Layers))
            {
                var layerType = layers.First.GetLayerType();

                if (layerType is LayerType.Dense)
                {
                    childLayers.Add(await (layers.First as Dense).Crossover(layers.Second as Dense, neatEnv, crossoverRate));
                }

                if (layerType is LayerType.LSTM)
                {
                    childLayers.Add(await (layers.First as LSTM).Crossover(layers.Second as LSTM, neatEnv, crossoverRate));
                }
            }

            return new Neat()
            {
                Layers = childLayers
            } as T;
        }

        public override async Task<double> Distance<T, TE>(T other, TE environment)
        {
            var parentTwo = other as Neat;
            var neatEnv = environment as NeatEnvironment;

            var sum = 0.0;
            foreach (var layers in Layers.Zip(parentTwo.Layers))
            {
                sum += layers.First.GetLayerType() switch
                {
                    LayerType.Dense => await (layers.First as Dense).Distance(layers.Second as Dense, neatEnv),
                    LayerType.LSTM => await (layers.First as LSTM).Distance(layers.Second as LSTM, neatEnv),
                    _ => throw new KeyNotFoundException($"{layers.First.GetType()} is not implemented")
                };
            }

            return sum;
        }


        public override T CloneGenome<T>() => new Neat()
        {
            Layers = Layers.Select(layer => layer.CloneLayer()).ToList()
        } as T;

        public override void ResetGenome()
        {
            foreach (var layer in Layers)
            {
                layer.Reset();
            }
        }
    }
}