
namespace Radiate.Optimizers.Evolution.Environment;

public abstract class EvolutionEnvironment
{
    public abstract void Reset();
    public virtual string ToJson() => "";
}

public class BaseEvolutionEnvironment : EvolutionEnvironment
{
    public override void Reset() { }
}
