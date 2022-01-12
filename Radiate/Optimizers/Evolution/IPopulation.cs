namespace Radiate.Optimizers.Evolution;

public interface IPopulation
{
    Task<float> Step();
}