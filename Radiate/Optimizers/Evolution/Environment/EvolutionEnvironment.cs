
namespace Radiate.Optimizers.Evolution.Environment;

public abstract class EvolutionEnvironment
{
    public abstract void Reset();
    public virtual string ToJson() => "";
    public abstract T GenerateGenome<T>() where T: class, IGenome;
}

public class BaseEvolutionEnvironment : EvolutionEnvironment
{
    public override void Reset() { }
    public override T GenerateGenome<T>() => null;
}
