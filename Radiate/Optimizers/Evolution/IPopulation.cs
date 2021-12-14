using System.Threading.Tasks;
using Radiate.Optimizers.Evolution.Engine;
using Radiate.Optimizers.Evolution.Engine.Delegates;

namespace Radiate.Optimizers.Evolution
{
    public interface IPopulation
    {
        Task<Member<Genome>> Evolve(Genome genome, Run runFunction);
    }
}