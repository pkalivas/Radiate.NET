using Radiate.Records;

namespace Radiate.Optimizers.Evolution.Info;

public record PopulationInfo<T>(
    PopulationSettings PopulationSettings = null,
    EvolutionEnvironment EvolutionEnvironment = null,
    FitnessFunction<T> FitnessFunc = null) where T : class
{
    public PopulationInfo<T> AddSettings(Action<PopulationSettings> settings)
    {
        var currentSettings = PopulationSettings ?? new();
        settings.Invoke(currentSettings);
        return this with { PopulationSettings = currentSettings };
    }

    public PopulationInfo<T> AddEnvironment(EvolutionEnvironment environment) => 
        this with { EvolutionEnvironment = environment };

    public PopulationInfo<T> AddFitnessFunction(FitnessFunction<T> solver) =>
        this with { FitnessFunc = solver };

    public DistanceControl DistanceControl =>
        new(PopulationSettings.COne, PopulationSettings.CThree, PopulationSettings.CThree);

    public StagnationControl StagnationControl => new(PopulationSettings.StagnationLimit);

    public PassDownControl PassDownControl =>
        new(PopulationSettings.InbreedRate, PopulationSettings.CrossoverRate, 0);
}