
namespace Radiate.NET.Optimizers.Evolution.Engine
{
    public abstract class EvolutionEnvironment
    {
        public abstract void Reset();
    }

    public class BaseEvolutionEnvironment : EvolutionEnvironment
    {
        public override void Reset() { }
    }
}