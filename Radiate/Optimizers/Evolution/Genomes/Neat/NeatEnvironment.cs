using Radiate.Activations;

namespace Radiate.Optimizers.Evolution.Genomes.Neat;

public class NeatEnvironment : EvolutionEnvironment
{
    public int InputSize { get; set; }
    public int OutputSize { get; set; }
    public float ReactivateRate { get; set; }
    public float WeightMutateRate { get; set; }
    public float NewNodeRate { get; set; }
    public float NewEdgeRate { get; set; }
    public float EditWeights { get; set; }
    public float WeightPerturb { get; set; }
    public float RecurrentNeuronRate { get; set; }
    public Activation OutputLayerActivation { get; set; }
    public List<Activation> ActivationFunctions { get; set; }

    public override void Reset() { }

    public override T GenerateGenome<T>()
    {
        return new Neat(InputSize, OutputSize, OutputLayerActivation) as T;
    }
}
