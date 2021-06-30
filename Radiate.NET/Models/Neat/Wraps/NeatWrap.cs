using System;
using System.Collections.Generic;
using Radiate.NET.Models.Neat.Enums;
using Radiate.NET.Models.Neat.Structs;

namespace Radiate.NET.Models.Neat.Wraps
{
    public class NeatWrap
    {
        public int LayerCount { get; set; }
        public List<LayerWrap> LayerWraps { get; set; }
    }

    public class LayerWrap
    {
        public LayerType LayerType { get; set; }
        public DenseWrap Dense { get; set; }
    }

    public class DenseWrap
    {
        public NeuronId[] Inputs { get; set; }
        public NeuronId[] Outputs { get; set; }
        public List<Neuron> Nodes { get; set; }
        public List<Edge> Edges { get; set; }
        public Dictionary<Guid, EdgeId> EdgeInnovationLookup { get; set; }
        public ActivationFunction Activation { get; set; }
        public LayerType LayerType { get; set; }
        public bool FastMode { get; set; }
    }


}