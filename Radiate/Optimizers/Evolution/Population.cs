using Radiate.Optimizers.Evolution.Info;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public class Population<T> : IPopulation where T : class, IGenome
{
    private const double DistanceMultiplier = 100.0;
    private const double Tolerance = 0.0000001;

    private PopulationControl _populationControl;
    private StagnationControl _stagnationControl;
    private Generation _generation;
    private readonly PassDownControl _passDownControl;
    private readonly EvolutionEnvironment _evolutionEnvironment;
    private readonly Solve<T> _fitnessFunction;
    
    private Generation CurrentGeneration { get; set; }
    private InnovationCounter InnovationCounter { get; set; } = new();

    public Population() { }

    public Population(PopulationInfo<T> info) : this(info, new List<T>()) { }
    
    public Population(T genome) : this(new(), new []{ genome }) { }

    public Population(PopulationInfo<T> info, T genome) : this(info, new[] { genome }) { }
    
    public Population(PopulationInfo<T> info, IEnumerable<T> genomes)
    {
        var popSettings = info.PopulationSettings ?? new();
        
        _populationControl = new(popSettings.DynamicDistance, popSettings.SpeciesDistance, popSettings.SpeciesTarget);
        _stagnationControl = new(popSettings.CleanPct, popSettings.StagnationLimit);
        _passDownControl = new(popSettings.InbreedRate, popSettings.CrossoverRate, popSettings.Size);
        _fitnessFunction = info.FitnessFunc;
        _evolutionEnvironment = info.EvolutionEnvironment;
        _generation = CreateSeedGeneration(genomes);
    }
    

    // public Population(T genome)
    // {
    //     // CurrentGeneration = new Generation
    //     // {
    //     //     Members = new Dictionary<Guid, Member>
    //     //     {
    //     //         { Guid.NewGuid(), new Member { Fitness = 0, Model = genome } }
    //     //     }
    //     // };
    // }

    public Population(IEnumerable<IGenome> genomes)
    {
        // PopulationSettings = new PopulationSettings(genomes.Count());
        // CurrentGeneration = new Generation
        // {
        //     Members = genomes
        //         .Select(member => (
        //             Id: Guid.NewGuid(),
        //             Member: new Member
        //             {
        //                 Fitness = 0,
        //                 Model = member
        //             }
        //         ))
        //         .ToDictionary(key => key.Id, val => val.Member),
        //     Species = new List<Niche>()
        // };
    }
    
    public Population<T> AddFitnessFunction(Solve<T> solver)
    {
        // Solver = solver;
        return this;
    }

    public Population<T> AddSettings(Action<PopulationSettings> settings) 
    {
        // settings.Invoke(PopulationSettings);
        //
        // if (CurrentGeneration is null && EvolutionEnvironment is not null)
        // {
        //     DelayedInit<T>();
        // }
        //
        return this;
    }

    public Population<T> AddEnvironment(EvolutionEnvironment environment)
    {
        // EvolutionEnvironment = environment;
        //
        // if (CurrentGeneration is null && PopulationSettings is not null)
        // {
        //     DelayedInit<T>();
        // }
        //
        return this;
    }

    public async Task<Generation> Evolve(int index)
    {
        // if (CurrentGeneration is null || EvolutionEnvironment is null)
        // {
        //     throw new Exception($"Cannot evolve a generation with no EvolutionEnvironment");
        // }
        
        await _generation.Step(_fitnessFunction, _populationControl, _evolutionEnvironment);
        
        return _generation;
    }

    public float PassDown()
    {
        var topMember = _generation.GetBestMember();

        _populationControl = AdjustDistance();
        _stagnationControl = AdjustStagnation(topMember.Fitness);

        _generation = _generation.CreateNextGeneration(_passDownControl, _evolutionEnvironment);

        return topMember.Fitness;
    }

    public Generation CreateSeedGeneration(IEnumerable<IGenome> genomes)
    {
        return new Generation
        {
            Members = (genomes.Any() ? DefaultGenomes() : genomes)
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

    IGenome IPopulation.Best() => CurrentGeneration.GetBestMember().Model;

    private StagnationControl AdjustStagnation(float fitness)
    {
        var (cleanPercent, stagnationLimit, stagnationCount, previousFitness) = _stagnationControl;
        var newStagnationControl = _stagnationControl with { };
        
        if (stagnationCount >= stagnationLimit)
        {
            CurrentGeneration.CleanPopulation(cleanPercent);
            newStagnationControl = _stagnationControl with { StagnationCount = 0 };
        }
        else if (Math.Abs(previousFitness - fitness) < Tolerance)
        {
            newStagnationControl = _stagnationControl with { StagnationCount = stagnationCount + 1 };
        }
        else
        {
            newStagnationControl = _stagnationControl with { StagnationCount = 0 };
        }

        return newStagnationControl with { PreviousFitness = fitness };
    }
    
    private PopulationControl AdjustDistance()
    {
        var (dynamicDistance, distance, target, precision) = _populationControl;
        var newControl = _populationControl with { };

        if (!dynamicDistance)
        {
            return newControl;
        }
        
        if (CurrentGeneration.Species.Count < target)
        {
            var newDistance = distance - precision;
            while (newDistance <= 0)
            {
                precision *= DistanceMultiplier;
                newDistance = distance - precision;
            }

            newControl = newControl with
            {
                Distance = Math.Round(newDistance, 5)
            };
        }
        else if (CurrentGeneration.Species.Count > target)
        {
            var newDistance = distance + precision;

            newControl = newControl with
            {
                Distance = Math.Round(newDistance, 5)
            };
        }

        return newControl;
    }
    
    private IEnumerable<T> DefaultGenomes() => Enumerable.Range(0, _passDownControl.Size)
        .Select(_ => _evolutionEnvironment.GenerateGenome<T>())
        .ToList();

    private Generation DelayedInit<T>() where T : class, IGenome
    {
        var genomes = Enumerable.Range(0, _passDownControl.Size)
            .Select(_ => _evolutionEnvironment.GenerateGenome<T>())
            .ToList();

        return new Generation
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