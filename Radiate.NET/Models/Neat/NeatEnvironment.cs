using System.Collections.Generic;
using Radiate.NET.Engine;
using Radiate.NET.Models.Neat.Enums;

namespace Radiate.NET.Models.Neat
{
    public class NeatEnvironment : EvolutionEnvironment
    {
        public float ReactivateRate { get; set; }
        public float WeightMutateRate { get; set; }
        public float NewNodeRate { get; set; }
        public float NewEdgeRate { get; set; }
        public float EditWeights { get; set; }
        public float WeightPerturb { get; set; }
        public List<ActivationFunction> ActivationFunctions { get; set; }

        public override void Reset() { }
    }
}