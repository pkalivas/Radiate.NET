using System.Threading.Tasks;
using Radiate.NET.Optimizers.Evolution.Engine;
using Radiate.NET.Optimizers.Evolution.Engine.Delegates;

namespace Radiate.NET.Optimizers
{
    public interface IPopulation
    {
        Task<Member<Genome>> Evolve(Genome genome, Run runFunction);
    }
}