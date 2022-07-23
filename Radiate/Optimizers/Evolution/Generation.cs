using System.Collections.Concurrent;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Records;

namespace Radiate.Optimizers.Evolution;

public class Generation
{
    private Dictionary<Guid, Member> Members { get; set; }
    private Dictionary<Guid, Member> MascotMembers { get; set; }
    private List<Niche> Species { get; set; }
    private PopulationControl _populationControl;
    private StagnationControl _stagnationControl;
    private readonly PassDownControl _passDownControl;
    private readonly EvolutionEnvironment _evolutionEnvironment;
    
    private int _generationNumber = 0;

    public Generation(Dictionary<Guid, Member> members, PopulationSettings popSettings, EvolutionEnvironment evolutionEnvironment)
    {
        Members = members;
        Species = new List<Niche>();
        MascotMembers = new();
        _populationControl = new(popSettings.DynamicDistance, popSettings.SpeciesDistance, popSettings.SpeciesTarget,
            popSettings.COne, popSettings.CTwo, popSettings.CThree);
        _stagnationControl = new(popSettings.CleanPct, popSettings.StagnationLimit);
        _passDownControl = new(popSettings.InbreedRate, popSettings.CrossoverRate, popSettings.Size);
        _evolutionEnvironment = evolutionEnvironment;
    }
    
    public async Task<Generation> Step<T>(Solve<T> problem) where T : class
    {
        var newMembers = new ConcurrentBag<(Guid memberId, float memberFitness)>();
        Parallel.ForEach(Members.Keys, member =>
        {
            newMembers.Add((member, problem(Members[member].Model as T)));
        });
        
        foreach (var (key, val) in newMembers)
        {
            if (Members.ContainsKey(key))
            {
                Members[key].Fitness = val;
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
        var newMembers = SurvivorSelector.Select(Members, Species)
            .Select(pair =>
            {
                var model = pair.member.Model;
                model.ResetGenome();
                return (pair.memberId, member: new Member { Fitness = 0, Model = model });
            })
            .ToDictionary(key => key.memberId, val => val.member);
        
        var childNum = size - newMembers.Count;
        var childTasks = new ConcurrentBag<IGenome>();
        Parallel.For(0, childNum, _ =>
        {
            var (parentOneId, parentTwoId) = ParentSelector.Select(inbreedRate, Species);
            var parentOne = Members[parentOneId];
            var parentTwo = Members[parentTwoId];
        
            var newGenomeTask = parentOne.Fitness > parentTwo.Fitness 
                ? parentOne.Model.Crossover(parentTwo.Model, _evolutionEnvironment, crossoverRate) 
                : parentTwo.Model.Crossover(parentOne.Model, _evolutionEnvironment, crossoverRate);
            
            childTasks.Add(newGenomeTask);
        });

        foreach (var child in childTasks)
        {
            newMembers[Guid.NewGuid()] = new Member { Fitness = 0, Model = child };
        }
        
        Species.ForEach(niche => niche.Reset());
        MascotMembers = Species
            .Select(spec => (spec.Mascot, Members[spec.Mascot]))
            .ToDictionary(key => key.Mascot, val => val.Item2);
        Members = newMembers;
        _generationNumber++;
    }

    private void CleanPopulation(double pct)
    {
        var toRemove = new List<Guid>();
        foreach (var species in Species)
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
            Members.Remove(memId);
        }
    }

    public Member GetBestMember() => 
        Members.Values
            .Aggregate(Members.Values.First(), (best, current) => current.Fitness > best.Fitness ? current : best);

    public GenerationReport GetReport() => new()
    {
        GenerationNum = _generationNumber,
        NumMembers = Members.Count,
        NumNiche = Species.Count,
        TopFitness = GetBestMember().Fitness,
        StagnationCount = _stagnationControl.StagnationCount,
        Distance = _populationControl.Distance,
        NicheReports = Species
            .Select(spec => spec.GetReport())
            .OrderBy(val => val.Age)
            .ToList()
    };
    
    private async Task Speciate()
    {
        var retainedSpecies = new HashSet<Guid>();

        foreach (var (id, member) in Members)
        {
            var found = false;
            foreach (var species in Species)
            {
                var speciesMember = MascotMembers[species.Mascot].Model;
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
                Species.Add(newSpecies);
                retainedSpecies.Add(newSpecies.NicheId);
                MascotMembers[id] = member;
            }
        }

        foreach (var species in Species.Where(species => !retainedSpecies.Contains(species.NicheId)))
        {
            MascotMembers.Remove(species.NicheId);
        }

        Species = Species.Where(spec => retainedSpecies.Contains(spec.NicheId)).ToList();

        foreach (var species in Species)
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
        
        if (Species.Count < target)
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
        else if (Species.Count > target)
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