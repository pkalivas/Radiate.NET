using Radiate.Domain.Activation;
using Radiate.Optimizers.Evolution.Population;

namespace Radiate.Optimizers.Evolution.Neat;

public class NeatEnvironment : EvolutionEnvironment
{
    public float ReactivateRate { get; set; }
    public float WeightMutateRate { get; set; }
    public float NewNodeRate { get; set; }
    public float NewEdgeRate { get; set; }
    public float EditWeights { get; set; }
    public float WeightPerturb { get; set; }
    public List<Activation> ActivationFunctions { get; set; }

    public override void Reset() { }
}
