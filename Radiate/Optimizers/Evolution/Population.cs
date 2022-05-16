

using Radiate.Optimizers.Evolution.Environment;

namespace Radiate.Optimizers.Evolution;

public class Population<T> : IPopulation where T : class
{
    private PopulationSettings Settings { get; set; }
    private Generation CurrentGeneration { get; set; }
    private EvolutionEnvironment EvolutionEnvironment { get; set; }
    private Solve<T> Solver { get; set; }
    private int GenerationsUnchanged { get; set; }
    private double PreviousFitness { get; set; }
    private const double Tolerance = 0.0000001;


    public Population(IEnumerable<IGenome> genomes)
    {
        Settings = new PopulationSettings(genomes.Count());
        CurrentGeneration = new Generation
        {
            Members = genomes
                .Select(member => (
                    Id: Guid.NewGuid(),
                    Member: new Member
                    {
                        Fitness = 0,
                        Model = member
                    }
                ))
                .ToDictionary(key => key.Id, val => val.Member),
            Species = new List<Niche>()
        };
    }

    public async Task<float> Step()
    {
        var topMember = CurrentGeneration.Step(Solver);

        await CurrentGeneration.Speciate(Settings.SpeciesDistance, EvolutionEnvironment);
        
        if (Settings.DynamicDistance)
        {
            if (CurrentGeneration.Species.Count < Settings.SpeciesTarget)
            {
                Settings.SpeciesDistance -= .1;
            }
            else if (CurrentGeneration.Species.Count > Settings.SpeciesTarget)
            {
                Settings.SpeciesDistance += .1;
            }
        
            if (Settings.SpeciesDistance < .01)
            {
                Settings.SpeciesDistance = .01;
            }
        }
        
        if (GenerationsUnchanged >= Settings.StagnationLimit)
        {
            CurrentGeneration.CleanPopulation(Settings.CleanPct);
            GenerationsUnchanged = 0;
        }
        else if (Math.Abs(PreviousFitness - topMember.Fitness) < Tolerance)
        {
            GenerationsUnchanged++;
        }
        else
        {
            GenerationsUnchanged = 0;
        }

        CurrentGeneration = CurrentGeneration.CreateNextGeneration(Settings, EvolutionEnvironment);
        
        PreviousFitness = topMember.Fitness;

        return topMember.Fitness;
    }

    IGenome IPopulation.Best() => CurrentGeneration.GetBestMember().Model;

    public Population<T> AddFitnessFunction(Solve<T> solver)
    {
        Solver = solver;
        return this;
    }

    public Population<T> AddSettings(Action<PopulationSettings> settings)
    {
        settings.Invoke(Settings);
        return this;
    }

    public Population<T> AddEnvironment(EvolutionEnvironment environment)
    {
        EvolutionEnvironment = environment;
        return this;
    }
}