using Radiate.Activations;
using Radiate.Optimizers.Evolution.Environment;

namespace Radiate.Optimizers.Evolution.Neat;

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
        var random = new Random();
        return new Neat(InputSize, OutputSize, OutputLayerActivation) as T;
    }
}
