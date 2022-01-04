using Radiate.Domain.Records;
using Radiate.Optimizers.Evolution.Population;
using Radiate.Optimizers.Evolution.Population.Delegates;
using Radiate.Optimizers.Evolution.Population.ParentalCriteria;
using Radiate.Optimizers.Evolution.Population.SurvivorCriteria;

namespace Radiate.Optimizers.Evolution;

 public class Population<T, TE> : IPopulation
        where T: Genome
        where TE: EvolutionEnvironment
{
    private PopulationSettings Settings { get; set; }
    private Generation<T, TE> CurrentGeneration { get; set; }
    private EvolutionEnvironment EvolutionEnvironment { get; set; }
    private Solve<T> Solver { get; set; }

    private GetSurvivors<T> SurvivorPicker { get; set; } = new Fittest().Pick<T>;
    private GetParents<T> ParentPicker { get; set; } = new BiasedRandom().Pick<T>;
    private int GenerationsUnchanged { get; set; }
    private double PreviousFitness { get; set; }
    private const double Tolerance = 0.0000001;


    public Population(IEnumerable<T> genomes)
    {
        Settings = new PopulationSettings();
        CurrentGeneration = new Generation<T, TE>
        {
            Members = genomes
                .Select(member => (
                    Id: Guid.NewGuid(),
                    Member: new Member<T>
                    {
                        Fitness = 0,
                        Model = member
                    }
                ))
                .ToDictionary(key => key.Id, val => val.Member),
            Species = new List<Niche>()
        };
    }
    
    public async Task Evolve(Func<Epoch, bool> trainFunc)
    {
        var count = 0;
        while (true)
        {
            var topMember = await EvolveGeneration();
            var epoch = new Epoch(count++, 0, topMember.Fitness, topMember.Fitness);

            if (trainFunc(epoch))
            {
                break;
            }
        }
    }

    public T Best => CurrentGeneration.GetBestMember().Model;

    public Population<T, TE> SetFitnessFunction(Solve<T> solver)
    {
        Solver = solver;
        return this;
    }

    public Population<T, TE> Configure(Action<PopulationSettings> settings)
    {
        settings.Invoke(Settings);
        return this;
    }

    public Population<T, TE> SetSurvivorPicker(GetSurvivors<T> survivors)
    {
        SurvivorPicker = survivors;

        return this;
    }

    public Population<T, TE> SetParentPicker(GetParents<T> parents)
    {
        ParentPicker = parents;

        return this;
    }

    public Population<T, TE> SetEnvironment(EvolutionEnvironment environment)
    {
        EvolutionEnvironment = environment;

        return this;
    }

    private async Task<Member<T>> EvolveGeneration()
    {
        await CurrentGeneration.Optimize(Solver);

        var topMember = CurrentGeneration.GetBestMember();

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
        
        PreviousFitness = topMember.Fitness;

        CurrentGeneration = await CurrentGeneration.CreateNextGeneration(Settings, EvolutionEnvironment, SurvivorPicker, ParentPicker);

        return topMember;
    }
    
}