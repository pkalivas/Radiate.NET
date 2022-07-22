namespace Radiate.Optimizers.Evolution.Info;

public record PopulationInfo<T>(
    PopulationSettings PopulationSettings = null,
    EvolutionEnvironment EvolutionEnvironment = null,
    Solve<T> FitnessFunc = null) where T : class
{
    public PopulationInfo<T> AddSettings(Action<PopulationSettings> settings)
    {
        settings.Invoke(PopulationSettings);
        return this with { PopulationSettings = PopulationSettings };
    }

    public PopulationInfo<T> AddEnvironment(EvolutionEnvironment environment) => 
        this with { EvolutionEnvironment = environment };

    public PopulationInfo<T> AddFitnessFunction(Solve<T> solver) =>
        this with { FitnessFunc = solver };
}