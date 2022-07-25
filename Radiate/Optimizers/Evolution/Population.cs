using Radiate.Optimizers.Evolution.Info;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public class Population<T> : IPopulation where T : class, IGenome
{
    private readonly Generation<T> _generation;

    private Gene _topMember;
    
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
        var generationBest = _generation.Step();

        if (_topMember is null || generationBest.Fitness > _topMember.Fitness)
        {
            _topMember = generationBest;
        }
        
        return _generation.GetReport();
    } 

    public float PassDown()
    {
        _generation.CreateNextGeneration();
        return _topMember.Fitness;
    }

    IGenome IPopulation.Best() => _topMember.Genome;
    
    private static Generation<T> CreateSeedGeneration(IEnumerable<IGenome> genomes, PopulationInfo<T> info)
    {
        var population = genomes.Any()
            ? genomes
            : new[] { info.EvolutionEnvironment.GenerateGenome<T>() };

        var members = population
            .Select(member => (
                Id: Guid.NewGuid(),
                Member: new GenomeFitnessPair(member) { Fitness = 0 }
            ))
            .ToDictionary(key => key.Id, val => val.Member);

        return new Generation<T>(members, info);
    }
}