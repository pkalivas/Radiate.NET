using System.Collections.Generic;
using Radiate.NET.Domain.Activation;
using Radiate.NET.Optimizers.Evolution.Engine;

namespace Radiate.NET.Optimizers.Evolution.Neat
{
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
}