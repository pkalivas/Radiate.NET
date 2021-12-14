
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Radiate.NET.Optimizers.Evolution.Engine
{
    public abstract class Genome
    {
        public abstract Task<Genome> Crossover(Genome other, EvolutionEnvironment environment, double crossoverRate);

        public abstract Task<double> Distance(Genome other, EvolutionEnvironment environment);
        
        public abstract float[] Forward(float[] data);

        public abstract Genome CloneGenome();

        public abstract void ResetGenome();
    }
}