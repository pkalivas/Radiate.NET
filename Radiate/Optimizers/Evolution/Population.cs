

using System.Globalization;
using Radiate.Optimizers.Evolution.Environment;

namespace Radiate.Optimizers.Evolution;

public class Population<T> : IPopulation where T : class, IOptimizerModel, IGenome
{
    private PopulationSettings Settings { get; set; } = new();
    private Generation CurrentGeneration { get; set; }
    private EvolutionEnvironment EvolutionEnvironment { get; set; }
    private Solve<T> Solver { get; set; }
    
    private int GenerationsUnchanged { get; set; }
    private double PreviousFitness { get; set; }
    private double DistanceMultiplier { get; set; } = 100.0;
    private const double Tolerance = 0.0000001;

    public Population() { }

    public Population(T genome)
    {
        CurrentGeneration = new Generation
        {
            Members = new Dictionary<Guid, Member>
            {
                { Guid.NewGuid(), new Member { Fitness = 0, Model = genome } }
            }
        };
    }

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
    
    public Population<T> AddFitnessFunction(Solve<T> solver)
    {
        Solver = solver;
        return this;
    }

    public Population<T> AddSettings(Action<PopulationSettings> settings) 
    {
        settings.Invoke(Settings);

        if (CurrentGeneration is null && EvolutionEnvironment is not null)
        {
            DelayedInit<T>();
        }
        
        return this;
    }

    public Population<T> AddEnvironment(EvolutionEnvironment environment)
    {
        EvolutionEnvironment = environment;
        
        if (CurrentGeneration is null && Settings is not null)
        {
            DelayedInit<T>();
        }
        
        return this;
    }

    public async Task<float> Step()
    {
        if (CurrentGeneration is null || EvolutionEnvironment is null)
        {
            throw new Exception($"Cannot evolve a generation with no EvolutionEnvironment");
        }
        
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

    private void AdjustDistance()
    {
        var adjustment = 10.0 / DistanceMultiplier;
        var newDistance = Settings.SpeciesDistance - adjustment;

        if (newDistance < 0)
        {
            DistanceMultiplier *= 10.0;
            adjustment = 10.0 / DistanceMultiplier;
            newDistance = Settings.SpeciesDistance - adjustment;
        }

        if (newDistance < 0)
        {
            throw new Exception($"Cannot move species distance below 0. Multiplier " +
                                $"{DistanceMultiplier} Adjustment {adjustment} Distance {newDistance}");
        }
        
        if (CurrentGeneration.Species.Count < Settings.SpeciesTarget)
        {
            Settings.SpeciesDistance -= adjustment;
        }
        else if (CurrentGeneration.Species.Count > Settings.SpeciesTarget)
        {
            Settings.SpeciesDistance += adjustment;
        }
    }

    private void DelayedInit<T>() where T : class, IGenome
    {
        var genomes = Enumerable.Range(0, Settings.Size)
            .Select(_ => EvolutionEnvironment.GenerateGenome<T>())
            .ToList();

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
}