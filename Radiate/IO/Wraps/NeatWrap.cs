using Radiate.Activations;
using Radiate.Optimizers.Evolution.Genomes.Neat;

namespace Radiate.IO.Wraps;

public class NeatWrap
{
    public NeuronId[] Inputs { get; set; }
    public NeuronId[] Outputs { get; set; }
    public List<Neuron> Nodes { get; set; }
    public List<Edge> Edges { get; set; }
    public Dictionary<Guid, EdgeId> EdgeInnovationLookup { get; set; }
    public Activation Activation { get; set; }
}