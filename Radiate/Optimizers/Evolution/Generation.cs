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
    
    private int _generationNumber = 0;

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
        AdjustStagnation();

        var (inbreedRate, crossoverRate, size) = _passDownControl;
        var newMembers = SurvivorSelector.Select(_members, _species)
            .Select(pair =>
            {
                var model = pair.member.Model;
                model.ResetGenome();
                return pair with { member = new Member { Fitness = 0, Model = model } };
            })
            .ToDictionary(key => key.memberId, val => val.member);
        
        var childNum = size - newMembers.Count;
        var childTasks = new ConcurrentBag<IGenome>();
        Parallel.For(0, childNum, _ =>
        {
            var (parentOneId, parentTwoId) = ParentSelector.Select(inbreedRate, _species);
            var parentOne = _members[parentOneId];
            var parentTwo = _members[parentTwoId];
        
            var newGenomeTask = parentOne.Fitness > parentTwo.Fitness 
                ? parentOne.Model.Crossover(parentTwo.Model, _evolutionEnvironment, crossoverRate) 
                : parentTwo.Model.Crossover(parentOne.Model, _evolutionEnvironment, crossoverRate);
            
            childTasks.Add(newGenomeTask);
        });

        foreach (var child in childTasks)
        {
            newMembers[Guid.NewGuid()] = new Member { Fitness = 0, Model = child };
        }
        
        _species.ForEach(niche => niche.Reset());
        _mascotMembers = _species
            .Select(spec => (spec.Mascot, _members[spec.Mascot]))
            .ToDictionary(key => key.Mascot, val => val.Item2);
        _members = newMembers;
        _generationNumber++;
    }

    private void CleanPopulation(double pct)
    {
        var toRemove = new List<Guid>();
        foreach (var species in _species)
        {
            if (species.Members.Count == 1)
            {
                continue;
            }

            var toTake = (int) Math.Ceiling(species.Members.Count - species.Members.Count * pct);

            species.Members = species.Members
                .OrderByDescending(mem => mem.fitness)
                .Take(toTake)
                .ToList();
            
            toRemove.AddRange(species.Members.Skip(toTake).Select(mem => mem.memberId));
        }

        foreach (var memId in toRemove)
        {
            _members.Remove(memId);
        }
    }

    public Member GetBestMember() => 
        _members.Values
            .Aggregate(_members.Values.First(), (best, current) => current.Fitness > best.Fitness ? current : best);

    public GenerationReport GetReport() => new()
    {
        GenerationNum = _generationNumber,
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
                    species.Members.Add((id, member.Fitness));
                    retainedSpecies.Add(species.NicheId);
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                var newSpecies = new Niche(id, member.Fitness);
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
    
    private void AdjustStagnation()
    {
        var (cleanPercent, stagnationLimit, stagnationCount, previousFitness) = _stagnationControl;
        var newStagnationControl = _stagnationControl with { };
        var topMember = GetBestMember();
        
        if (stagnationCount >= stagnationLimit)
        {
            CleanPopulation(cleanPercent);
            newStagnationControl = _stagnationControl with { StagnationCount = 0 };
        }
        else if (Math.Abs(previousFitness - topMember.Fitness) < EvolutionConstants.Tolerance)
        {
            newStagnationControl = _stagnationControl with { StagnationCount = stagnationCount + 1 };
        }
        else
        {
            newStagnationControl = _stagnationControl with { StagnationCount = 0 };
        }

        _stagnationControl = newStagnationControl with { PreviousFitness = topMember.Fitness };
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