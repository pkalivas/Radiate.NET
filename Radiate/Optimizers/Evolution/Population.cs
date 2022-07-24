using Radiate.Optimizers.Evolution.Info;
using Radiate.Optimizers.Evolution.Interfaces;

namespace Radiate.Optimizers.Evolution;

public class Population<T> : IPopulation where T : class, IGenome
{
    private readonly Generation<T> _generation;

    private Member _topMember;
    
    public Population(PopulationInfo<T> info) : this(info, new List<T>()) { }
    
    public Population(T genome) : this(new(), new[] { genome }) { }

    public Population(PopulationInfo<T> info, T genome) : this(info, new[] { genome }) { }
    
    public Population(PopulationInfo<T> info, IEnumerable<T> genomes)
    {
        InnovationCounter.Init();
        
        _generation = CreateSeedGeneration(genomes, info);
    }

    public async Task<GenerationReport> Evolve()
    {
        _topMember = await _generation.Step();
        return _generation.GetReport();
    } 

    public float PassDown()
    {
        _generation.CreateNextGeneration();
        return _topMember.Fitness;
    }

    IGenome IPopulation.Best() => _topMember.Model;
    
    private static Generation<T> CreateSeedGeneration(IEnumerable<IGenome> genomes, PopulationInfo<T> info)
    {
        var population = genomes.Any()
            ? genomes
            : new[] { info.EvolutionEnvironment.GenerateGenome<T>() };
            // : Enumerable.Range(0, info.populationSettings.Size!.Value)
            //     .Select(_ => evolutionEnvironment.GenerateGenome<T>())
            //     .ToList();

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

        return new Generation<T>(members, info);
    }
}