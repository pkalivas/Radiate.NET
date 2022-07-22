

using System.Globalization;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public class Population<T> : IPopulation where T : class, IGenome
{
    private PopulationSettings Settings { get; set; } = new();
    private Generation CurrentGeneration { get; set; }
    private EvolutionEnvironment EvolutionEnvironment { get; set; }
    private InnovationCounter InnovationCounter { get; set; } = new();
    private PopulationControl PopulationControl { get; set; }
    private Solve<T> Solver { get; set; }

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

    public async Task<float> Step(int index)
    {
        if (CurrentGeneration is null || EvolutionEnvironment is null)
        {
            throw new Exception($"Cannot evolve a generation with no EvolutionEnvironment");
        }

        if (index == 0)
        {
            PopulationControl = new PopulationControl(Settings.SpeciesDistance);
        }
        
        var topMember = CurrentGeneration.Step(Solver);

        await CurrentGeneration.Speciate(Settings.SpeciesDistance, EvolutionEnvironment);
        
        if (Settings.DynamicDistance)
        {
            AdjustDistance();
        }

        AdjustStagnation(topMember.Fitness);

        CurrentGeneration = CurrentGeneration.CreateNextGeneration(Settings, EvolutionEnvironment);

        return topMember.Fitness;
    }

    IGenome IPopulation.Best() => CurrentGeneration.GetBestMember().Model;

    private void AdjustStagnation(float fitness)
    {
        var (_, _, stagnationCount, prevFit) = PopulationControl;

        if (stagnationCount >= Settings.StagnationLimit)
        {
            CurrentGeneration.CleanPopulation(Settings.CleanPct);
            PopulationControl = PopulationControl with
            {
                StagnationCount = 0,
                PreviousFitness = prevFit
            };
        }
        else if (Math.Abs(prevFit - fitness) < Tolerance)
        {
            PopulationControl = PopulationControl with
            {
                StagnationCount = stagnationCount + 1, 
                PreviousFitness = fitness
            };
        }
        else
        {
            PopulationControl = PopulationControl with
            {
                StagnationCount = 0, 
                PreviousFitness = fitness
            };
        }
    }
    
    private void AdjustDistance()
    {
        var (distance, precision, _, _) = PopulationControl;

        if (CurrentGeneration.Species.Count < Settings.SpeciesTarget)
        {
            var newDistance = distance - precision;
            while (newDistance <= 0)
            {
                precision *= DistanceMultiplier;
                newDistance = distance - precision;
            }

            Settings.SpeciesDistance -= precision;
            PopulationControl = PopulationControl with
            {
                Distance = Math.Round(newDistance, 5)
            };
        }
        else if (CurrentGeneration.Species.Count > Settings.SpeciesTarget)
        {
            var newDistance = distance + precision;

            Settings.SpeciesDistance += precision;
            PopulationControl = PopulationControl with
            {
                Distance = Math.Round(newDistance, 5),
            };
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