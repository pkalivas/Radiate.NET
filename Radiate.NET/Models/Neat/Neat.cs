using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.NET.Engine;
using Radiate.NET.Enums;
using Radiate.NET.Models.Neat.Enums;
using Radiate.NET.Models.Neat.Layers;
using Radiate.NET.Models.Neat.Wraps;

namespace Radiate.NET.Models.Neat
{
    public class Neat : Genome, ITradingModel
    {
        public List<ILayer> Layers { get; set; }

        public Neat()
        {
            Layers = new List<ILayer>();
        }


        public Neat AddLayer(ILayer layer)
        {
            Layers.Add(layer);
            return this;
        }


        public List<float> Forward(List<float> data) =>
            Layers.Aggregate(data, (current, layer) => layer.Forward(current));


        public TradingModel Wrap() => new()
        {
            ModelType = ModelType.Neat,
            NeatWrap = new NeatWrap
            {
                LayerCount = Layers.Count,
                LayerWraps = Layers.Select(layer => layer.Wrap()).ToList()
            }
        };
        

        public (int inSize, int outSize) Shape()
        {
            if (!Layers.Any())
            {
                return (inSize: 0, outSize: 0);
            }

            return (Layers.First().Shape().inSize, Layers.Last().Shape().outSize);
        }


        public override async Task<T> Crossover<T, TE>(T other, TE environment, double crossoverRate)
        {
            var parentTwo = other as Neat;
            var neatEnv = environment as NeatEnvironment;

            var childLayers = new List<ILayer>();
            foreach (var layers in Layers.Zip(parentTwo.Layers))
            {
                var resultTask = layers.First.GetType() switch
                {
                    LayerType.Dense => (layers.First as Dense).Crossover(layers.Second as Dense, neatEnv, crossoverRate),
                    LayerType.DensePool => (layers.First as Dense).Crossover(layers.Second as Dense, neatEnv, crossoverRate),
                    _ => throw new KeyNotFoundException($"{layers.First.GetType()} is not implemented.")
                };

                childLayers.Add(await resultTask);
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
                sum += layers.First.GetType() switch
                {
                    LayerType.Dense => await (layers.First as Dense).Distance(layers.Second as Dense, neatEnv),
                    LayerType.DensePool => await (layers.First as Dense).Distance(layers.Second as Dense, neatEnv),
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