using System.Collections.Concurrent;
using Radiate.Optimizers.Evolution.Info;
using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public class Generation<T> where T : class
{
    private readonly ConcurrentDictionary<Guid, Member> _members;
    private readonly ConcurrentDictionary<Guid, Member> _mascotMembers;
    private readonly ConcurrentDictionary<Guid, Niche> _species;
    private readonly DistanceManager _distanceManager;
    private readonly DistanceControl _distanceControl;
    private readonly StagnationControl _stagnationControl;
    private readonly PassDownControl _passDownControl;
    private readonly EvolutionEnvironment _evolutionEnvironment;
    private readonly FitnessFunction<T> _fitnessFunction;

    public Generation(Dictionary<Guid, Member> members, PopulationInfo<T> info)
    {
        var popSettings = info.PopulationSettings ?? new();
        var environment = info.EvolutionEnvironment ?? new BaseEvolutionEnvironment();
        
        _members = new ConcurrentDictionary<Guid, Member>(members);
        _mascotMembers = new ConcurrentDictionary<Guid, Member>();
        _species = new ConcurrentDictionary<Guid, Niche>();
        _distanceManager = new DistanceManager(popSettings.DynamicDistance, popSettings.SpeciesTarget, popSettings.SpeciesDistance);
        _distanceControl = info.DistanceControl;
        _stagnationControl = info.StagnationControl;
        _passDownControl = info.PassDownControl with { Size = popSettings.Size ?? members.Count };
        _evolutionEnvironment = environment;
        _fitnessFunction = info.FitnessFunc;
    }
    
    public async Task<Member> Step()
    {
        var retainedSpecies = new ConcurrentBag<Guid>();
        var newSpeciesMemberIds = new ConcurrentBag<Guid>();
        
        Parallel.ForEach(_members.Keys, memberId =>
        {
            _members[memberId].Fitness = _fitnessFunction(_members[memberId].Model as T);
        });
        
        Parallel.ForEach(_members, member =>
        {
            var (memberId, model) = member;
            var found = false;
            foreach (var (nicheId, species) in _species)
            {
                var mascot = _mascotMembers[species.Mascot].Model;
                var distance = model.Model.Distance(mascot, _distanceControl);

                if (distance < _distanceManager.Distance)
                {
                    if (!retainedSpecies.Contains(nicheId))
                    {
                        retainedSpecies.Add(nicheId);
                    }
                    
                    species.AddMember(new NicheMember(memberId, model.Fitness));
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                newSpeciesMemberIds.Add(memberId);
            }
        });

        foreach (var newSpeciesMember in newSpeciesMemberIds)
        {
            var member = _members[newSpeciesMember];
            var nicheId = Guid.NewGuid();
            var newSpecies = new Niche(nicheId, new NicheMember(newSpeciesMember, member.Fitness), _stagnationControl);
            retainedSpecies.Add(nicheId);
            
            _species[nicheId] = newSpecies;
            _mascotMembers[newSpeciesMember] = member;
        }
        
        var failedSpecies = _species.Keys.Where(nicheId => !retainedSpecies.Contains(nicheId));
        foreach (var failedId in failedSpecies)
        {
            _mascotMembers.Remove(_species[failedId].Mascot, out _);
            _species.Remove(failedId, out _);
        }

        foreach (var species in _species.Values)
        {
            species.CalcTotalAdjustedFitness();
        }

        return GetBestMember();
    }
    
    public void CreateNextGeneration()
    {
        var (inbreedRate, crossoverRate, size) = _passDownControl;
        _distanceManager.Update(_species.Count);

        var survivors = _species.Values
            .Select(niche => niche.BestMember())
            .ToHashSet();

        var oldMemberIds = _members.Keys.Where(memberId => !survivors.Contains(memberId));

        foreach (var survivor in survivors.Select(survivorId => _members[survivorId]))
        {
            survivor.Model.ResetGenome();
            survivor.Fitness = 0;
        }

        var childNum = size - survivors.Count;
        Parallel.For(0, childNum, _ =>
        {
            var (parentOneId, parentTwoId) = ParentSelector.Select(inbreedRate, _species.Values);
            var parentOne = _members[parentOneId];
            var parentTwo = _members[parentTwoId];
        
            var newGenomeTask = parentOne.Fitness > parentTwo.Fitness 
                ? parentOne.Model.Crossover(parentTwo.Model, _evolutionEnvironment, crossoverRate) 
                : parentTwo.Model.Crossover(parentOne.Model, _evolutionEnvironment, crossoverRate);
        
            _members[Guid.NewGuid()] = new Member { Fitness = 0, Model = newGenomeTask };
        });

        foreach (var (_, species) in _species)
        {
            species.Reset();
            _mascotMembers[species.Mascot] = _members[species.Mascot];
        }

        foreach (var memberId in oldMemberIds)
        {
            _members.Remove(memberId, out _);
        }
    }

    private Member GetBestMember() => 
        _members.Values
            .Aggregate(_members.Values.First(), (best, current) => current.Fitness > best.Fitness ? current : best);

    public GenerationReport GetReport() => new()
    {
        NumMembers = _members.Count,
        NumNiche = _species.Count,
        TopFitness = GetBestMember().Fitness,
        StagnationCount = _stagnationControl.StagnationCount,
        Distance = _distanceManager.Distance,
        NicheReports = _species.Values
            .Select(spec => spec.GetReport())
            .OrderBy(val => val.Age)
            .ToList()
    };
}