using System.Collections.Concurrent;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public class Generation
{
    private Dictionary<Guid, Member> _members;
    private Dictionary<Guid, Member> _mascotMembers;
    private List<Niche> _species;
    private PopulationControl _populationControl;
    private StagnationControl _stagnationControl;
    private readonly PassDownControl _passDownControl;
    private readonly EvolutionEnvironment _evolutionEnvironment;
    
    public Generation(Dictionary<Guid, Member> members, PopulationSettings popSettings, EvolutionEnvironment evolutionEnvironment)
    {
        _members = members;
        _species = new List<Niche>();
        _mascotMembers = new();
        _populationControl = new(popSettings.DynamicDistance, popSettings.SpeciesDistance, popSettings.SpeciesTarget,
            popSettings.COne, popSettings.CTwo, popSettings.CThree);
        _stagnationControl = new(popSettings.CleanPct, popSettings.StagnationLimit);
        _passDownControl = new(popSettings.InbreedRate, popSettings.CrossoverRate, popSettings.Size ?? members.Count);
        _evolutionEnvironment = evolutionEnvironment;
    }
    
    public async Task<Generation> Step<T>(Solve<T> problem) where T : class
    {
        var newMembers = new ConcurrentBag<(Guid memberId, float memberFitness)>();
        Parallel.ForEach(_members.Keys, member =>
        {
            newMembers.Add((member, problem(_members[member].Model as T)));
        });
        
        foreach (var (key, val) in newMembers)
        {
            if (_members.ContainsKey(key))
            {
                _members[key].Fitness = val;
            }
        }

        await Speciate();

        return this;
    }
    
    public void CreateNextGeneration()
    {
        AdjustDistance();

        var (inbreedRate, crossoverRate, size) = _passDownControl;
        var children = new ConcurrentDictionary<Guid, Member>();
        
        foreach (var (memberId, member) in SurvivorSelector.Select(_members, _species))
        {
            var model = member.Model;
            model.ResetGenome();
            children[memberId] = new Member { Fitness = 0, Model = model };
        }

        var childNum = size - children.Count;
        Parallel.For(0, childNum, _ =>
        {
            var (parentOneId, parentTwoId) = ParentSelector.Select(inbreedRate, _species);
            var parentOne = _members[parentOneId];
            var parentTwo = _members[parentTwoId];
        
            var newGenomeTask = parentOne.Fitness > parentTwo.Fitness 
                ? parentOne.Model.Crossover(parentTwo.Model, _evolutionEnvironment, crossoverRate) 
                : parentTwo.Model.Crossover(parentOne.Model, _evolutionEnvironment, crossoverRate);

            children[Guid.NewGuid()] = new Member { Fitness = 0, Model = newGenomeTask };
        });
        
        _species.ForEach(niche => niche.Reset());
        _mascotMembers = _species
            .Select(spec => (spec.Mascot, _members[spec.Mascot]))
            .ToDictionary(key => key.Mascot, val => val.Item2);
        _members = children.ToDictionary(key => key.Key, val => val.Value);
    }

    public Member GetBestMember() => 
        _members.Values
            .Aggregate(_members.Values.First(), (best, current) => current.Fitness > best.Fitness ? current : best);

    public GenerationReport GetReport() => new()
    {
        NumMembers = _members.Count,
        NumNiche = _species.Count,
        TopFitness = GetBestMember().Fitness,
        StagnationCount = _stagnationControl.StagnationCount,
        Distance = _populationControl.Distance,
        NicheReports = _species
            .Select(spec => spec.GetReport())
            .OrderBy(val => val.Age)
            .ToList()
    };
    
    private async Task Speciate()
    {
        var retainedSpecies = new HashSet<Guid>();

        foreach (var (id, member) in _members)
        {
            var found = false;
            foreach (var species in _species)
            {
                var speciesMember = _mascotMembers[species.Mascot].Model;
                var memberDistance = await member.Model.Distance(speciesMember, _populationControl);

                if (memberDistance < _populationControl.Distance)
                {
                    species.AddMember(new(id, member.Fitness));
                    retainedSpecies.Add(species.NicheId);
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                var newSpecies = new Niche(new(id, member.Fitness), _stagnationControl);
                _species.Add(newSpecies);
                retainedSpecies.Add(newSpecies.NicheId);
                _mascotMembers[id] = member;
            }
        }

        foreach (var species in _species.Where(species => !retainedSpecies.Contains(species.NicheId)))
        {
            _mascotMembers.Remove(species.Mascot);
        }

        _species = _species.Where(spec => retainedSpecies.Contains(spec.NicheId)).ToList();

        foreach (var species in _species)
        {
            species.CalcTotalAdjustedFitness();
        }
    }
    
    private void AdjustDistance()
    {
        var (dynamicDistance, distance, target, _, _, _, precision) = _populationControl;
        var newControl = _populationControl with { };

        if (!dynamicDistance)
        {
            return;
        }
        
        if (_species.Count < target)
        {
            var newDistance = distance - precision;
            while (newDistance <= 0)
            {
                precision /= EvolutionConstants.DistanceMultiplier;
                newDistance = distance - precision;
            }

            newControl = newControl with
            {
                Distance = Math.Round(newDistance, 5)
            };
        }
        else if (_species.Count > target)
        {
            var newDistance = distance + precision;

            newControl = newControl with
            {
                Distance = Math.Round(newDistance, 5)
            };
        }

        _populationControl = newControl;
    }
}