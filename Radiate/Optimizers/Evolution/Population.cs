using Radiate.Optimizers.Evolution.Info;
using Radiate.Optimizers.Evolution.Interfaces;

namespace Radiate.Optimizers.Evolution;

public class Population<T> : IPopulation where T : class, IGenome
{
    private readonly Generation _generation;
    private readonly Solve<T> _fitnessFunction;

    private Member _topMember;
    
    public Population(PopulationInfo<T> info) : this(info, new List<T>()) { }
    
    public Population(T genome) : this(new(), new[] { genome }) { }

    public Population(PopulationInfo<T> info, T genome) : this(info, new[] { genome }) { }
    
    public Population(PopulationInfo<T> info, IEnumerable<T> genomes)
    {
        InnovationCounter.Init();

        var popSettings = info.PopulationSettings ?? new();
        var environment = info.EvolutionEnvironment ?? new BaseEvolutionEnvironment();
        
        _fitnessFunction = info.FitnessFunc;
        _generation = CreateSeedGeneration(genomes, popSettings, environment);
    }

    public async Task<Generation> Evolve() => await _generation.Step(_fitnessFunction);

    public float PassDown()
    {
        _topMember = _generation.GetBestMember();
        _generation.CreateNextGeneration();
        return _topMember.Fitness;
    }

    IGenome IPopulation.Best() => _topMember.Model;
    
    private static Generation CreateSeedGeneration(IEnumerable<IGenome> genomes, PopulationSettings popSettings, EvolutionEnvironment evolutionEnvironment)
    {
        var population = genomes.Any() 
            ? genomes 
            : Enumerable.Range(0, 1)
                .Select(_ => evolutionEnvironment.GenerateGenome<T>())
                .ToList();

        var members = population
            .Select(member => (
                Id: Guid.NewGuid(),
                Member: new Member
                {
                    Fitness = 0,
                    Model = member
                }
            ))
            .ToDictionary(key => key.Id, val => val.Member);

        return new Generation(members, popSettings, evolutionEnvironment);
    }
}