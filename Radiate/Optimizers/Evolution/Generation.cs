using System.Collections.Concurrent;
using Radiate.Optimizers.Evolution.Info;
using Radiate.Optimizers.Evolution.Managers;
using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public class Generation<T> where T : class
{
    private readonly ConcurrentDictionary<Guid, GenomeFitnessPair> _genomes;
    private readonly SpeciesManager _speciesManager;
    private readonly PassDownControl _passDownControl;
    private readonly EvolutionEnvironment _evolutionEnvironment;
    private readonly FitnessFunction<T> _fitnessFunction;
    
    public Generation(Dictionary<Guid, GenomeFitnessPair> members, PopulationInfo<T> info)
    {
        var popSettings = info.PopulationSettings ?? new();
        
        _genomes = new ConcurrentDictionary<Guid, GenomeFitnessPair>(members);
        _speciesManager = new SpeciesManager(info.DistanceControl, info.CompatibilityControl, info.StagnationControl);
        _passDownControl = info.PassDownControl with { Size = popSettings.Size ?? members.Count };
        _evolutionEnvironment = info.Environment;
        _fitnessFunction = info.FitnessFunction;
    }
    
    public Gene Step()
    {
        Parallel.ForEach(_genomes.Keys, memberId =>
        {
            _genomes[memberId].Fitness = _fitnessFunction(_genomes[memberId].Genome as T);
        });

        var unspeciated = _speciesManager.UpdateSpeciesMascots(_genomes);
        foreach (var genomeId in unspeciated)
        {
            var genome = _genomes[genomeId];
            var speciesId = _speciesManager.FindGenomeSpecies(genomeId, genome.Genome);

            if (speciesId is not null)
            {
                var newMember = new SpeciesMember(genomeId, genome.Fitness);
                _speciesManager.AddMember(speciesId.Value, newMember);
            }
            else
            {
                var gene = new Gene(genomeId, genome.Fitness, genome.Genome);
                _speciesManager.CreateNewSpecies(gene);
            }
        }

        _speciesManager.AdjustFitness();
        
        var topGenome = _genomes.MaxBy(genome => genome.Value.Fitness);
        return new Gene(topGenome.Key, topGenome.Value.Fitness, topGenome.Value.Genome);
    }
    
    public void CreateNextGeneration()
    {
        var (inbreedRate, crossoverRate, size) = _passDownControl;

        var survivors = _speciesManager.GetSurvivors();
        var oldMemberIds = _genomes.Keys.Where(memberId => !survivors.Contains(memberId));
        
        foreach (var survivor in survivors.Select(survivorId => _genomes[survivorId]))
        {
            survivor.Genome.ResetGenome();
            survivor.Fitness = 0;
        }

        var childNum = size - survivors.Count;
        Parallel.For(0, childNum, _ =>
        {
            var (parentOneId, parentTwoId) = ParentSelector.Select(inbreedRate, _speciesManager.Species);
            var parentOne = _genomes[parentOneId];
            var parentTwo = _genomes[parentTwoId];
        
            var childGenome = parentOne.Fitness > parentTwo.Fitness 
                ? parentOne.Genome.Crossover(parentTwo.Genome, _evolutionEnvironment, crossoverRate) 
                : parentTwo.Genome.Crossover(parentOne.Genome, _evolutionEnvironment, crossoverRate);
        
            _genomes[Guid.NewGuid()] = new GenomeFitnessPair(childGenome) { Fitness = 0 };
        });
        
        foreach (var memberId in oldMemberIds)
        {
            _genomes.Remove(memberId, out _);
        }
    }
    
    public GenerationReport GetReport() => new()
    {
        NumMembers = _genomes.Count,
        TopFitness = _genomes.MaxBy(val => val.Value.Fitness).Value.Fitness,
        SpeciesReport = _speciesManager.GetReport()
    };
}