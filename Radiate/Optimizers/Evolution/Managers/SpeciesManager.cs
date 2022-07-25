using System.Collections.Concurrent;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Records;

namespace Radiate.Optimizers.Evolution.Managers;

public class SpeciesManager
{
    private readonly ConcurrentDictionary<Guid, Species> _species;
    private readonly DistanceManager _distanceManager;
    private readonly CompatibilityManager _compatibilityManager;

    public SpeciesManager(DistanceControl distanceControl, CompatibilityControl compatibilityControl)
    {
        _species = new ConcurrentDictionary<Guid, Species>();
        _distanceManager = new DistanceManager(distanceControl);
        _compatibilityManager = new CompatibilityManager(compatibilityControl);
    }

    public ICollection<Species> Species => _species.Values;

    public double Distance => _compatibilityManager.Distance;

    public List<Guid> UpdateSpeciesMascots(ConcurrentDictionary<Guid, GenomeFitnessPair> genomes)
    {
        _distanceManager.Clear();
        
        var unspeciated = new ConcurrentBag<Guid>(genomes.Keys);
        foreach (var speciesId in _species.Keys)
        {
            var currentSpecies = _species[speciesId];
            var (mascotId, _, mascot) = currentSpecies.Mascot;
            var candidates = new ConcurrentBag<(Guid genomeId, double distance)>();
            Parallel.ForEach(unspeciated, genomeId =>
            {
                var genome = genomes[genomeId];
                var distance = _distanceManager.Distance(genomeId, mascotId, genome.Genome, mascot);
                candidates.Add((genomeId, distance));
            });

            var closestGenome = candidates.MinBy(cand => cand.distance);
            var mascotGenome = genomes[closestGenome.genomeId];
            
            _species[speciesId] = new Species(currentSpecies, new Gene(closestGenome.genomeId, mascotGenome.Fitness, mascotGenome.Genome));
            
            unspeciated.TryTake(out closestGenome.genomeId);
        }

        return unspeciated.ToList();
    }

    public Guid? FindGenomeSpecies(Guid genomeId, IGenome genome)
    {
        var candidates = new ConcurrentBag<(Guid SpeciesId, double Distance)>();
        Parallel.ForEach(_species, speciesPair =>
        {
            var (speciesId, species) = speciesPair;
            var (mascotId, _, mascot) = species.Mascot;
            var distance = _distanceManager.Distance(genomeId, mascotId, genome, mascot);

            if (distance < _compatibilityManager.Distance)
            {
                candidates.Add((speciesId, distance));
            }
        });

        if (candidates.Any())
        {
            return candidates.MinBy(cand => cand.Distance).SpeciesId;
        }

        return null;
    }
    
    public HashSet<Guid> GetSurvivors()
    {
        var stagnantSpecies = _species
            .Where(spec => spec.Value.IsStagnant)
            .Select(spec => spec.Key);

        if (_species.Count - stagnantSpecies.Count() > 0)
        {
            foreach (var speciesId in stagnantSpecies)
            {
                _species.TryRemove(speciesId, out var _);
            }   
        }

        _compatibilityManager.Update(_species.Count);
        
        return _species.Values
            .Select(niche => niche.BestMember())
            .ToHashSet();
    }

    public void AdjustFitness()
    {
        foreach (var species in _species.Values)
        {
            species.AdjustFitness();
        }
    }

    public void CreateNewSpecies(Gene newGene, StagnationControl stagnationControl)
    {
        var newSpeciesId = Guid.NewGuid();
        _species[newSpeciesId] = new Species(newSpeciesId, newGene, stagnationControl);
    }
    
    public void AddMember(Guid speciesId, SpeciesMember newMember)
    {
        _species[speciesId].AddMember(newMember);
    }

    public SpeciesReport GetReport() => new()
    {
        Distance = _compatibilityManager.Distance,
        NicheReports = _species.Values
            .Select(val => val.GetReport())
            .OrderBy(val => val.Age)
            .ToList()
    };
}