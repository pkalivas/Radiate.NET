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

    public PopulationInfo<T> AddEnvironment(Func<EvolutionEnvironment> environmentFunc) => 
        this with { EvolutionEnvironment = environmentFunc() };

    public PopulationInfo<T> AddFitnessFunction(FitnessFunction<T> solver) =>
        this with { FitnessFunc = solver };

    public PopulationSettings Population => PopulationSettings ?? new PopulationSettings();

    public EvolutionEnvironment Environment => EvolutionEnvironment ?? new BaseEvolutionEnvironment();

    public FitnessFunction<T> FitnessFunction =>
        FitnessFunc ?? (_ => 0f);

    public DistanceControl DistanceControl =>
        new(Population.COne, Population.CTwo, Population.CThree);

    public StagnationControl StagnationControl => new(Population.StagnationLimit);

    public PassDownControl PassDownControl => new(Population.InbreedRate, Population.CrossoverRate, 0);

    public CompatibilityControl CompatibilityControl => new(Population.DynamicDistance, Population.SpeciesTarget,
        Population.SpeciesDistance);
}