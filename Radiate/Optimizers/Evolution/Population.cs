using Radiate.Domain.Loss;
using Radiate.Domain.Models;
using Radiate.Domain.Records;
using Radiate.Optimizers.Evolution.Population;
using Radiate.Optimizers.Evolution.Population.Delegates;
using Radiate.Optimizers.Evolution.Population.ParentalCriteria;
using Radiate.Optimizers.Evolution.Population.SurvivorCriteria;

namespace Radiate.Optimizers.Evolution;

 public class Population<T, E> : IPopulation
        where T: Genome
        where E: EvolutionEnvironment
{
    private PopulationSettings Settings { get; set; }
    private Generation<T, E> CurrentGeneration { get; set; }
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
        CurrentGeneration = new Generation<T, E>
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
    
    public async Task Evolve(Func<double, int, bool> trainFunc)
    {
        var epoch = 0;
        while (true)
        {
            var topMember = await EvolveGeneration();

            if (trainFunc(topMember.Fitness, epoch))
            {
                break;
            }

            epoch++;
        }
    }

    public T Best => CurrentGeneration.GetBestMember().Model;

    public Population<T, E> SetFitnessFunction(Solve<T> solver)
    {
        Solver = solver;
        return this;
    }

    public Population<T, E> Configure(Action<PopulationSettings> settings)
    {
        settings.Invoke(Settings);
        return this;
    }

    public Population<T, E> SetSurvivorPicker(GetSurvivors<T> survivors)
    {
        SurvivorPicker = survivors;

        return this;
    }

    public Population<T, E> SetParentPicker(GetParents<T> parents)
    {
        ParentPicker = parents;

        return this;
    }

    public Population<T, E> SetEnvironment(EvolutionEnvironment environment)
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