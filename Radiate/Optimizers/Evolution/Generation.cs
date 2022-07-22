using System.Collections.Concurrent;
using System.Threading;
using Radiate.Optimizers.Evolution.Interfaces;

namespace Radiate.Optimizers.Evolution;

public class Generation
{
    public Dictionary<Guid, Member> Members { get; set; }
    public Dictionary<Guid, Member> MascotMembers { get; set; }
    public List<Niche> Species { get; set; }
    
    public Generation()
    {
        Members = new Dictionary<Guid, Member>();
        MascotMembers = new Dictionary<Guid, Member>();
        Species = new List<Niche>();
    }

    public async Task Speciate(double distance, EvolutionEnvironment settings)
    {
        var retainedSpecies = new HashSet<Guid>();

        foreach (var (id, member) in Members)
        {
            var found = false;
            foreach (var species in Species)
            {
                var speciesMember = MascotMembers[species.Mascot].Model;
                var memberDistance = await member.Model.Distance(speciesMember, settings);

                if (memberDistance < distance)
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
    
    public Member Step<T>(Solve<T> problem) where T : class
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

        return GetBestMember();
    }
    
    public Generation CreateNextGeneration(PopulationSettings popSettings, EvolutionEnvironment envSettings)
    {
        var newMembers = SurvivorSelector.Select(Members, Species)
            .Select(pair =>
            {
                var model = pair.member.Model;
                model.ResetGenome();
                return (pair.memberId, member: new Member { Fitness = 0, Model = model });
            })
            .ToDictionary(key => key.memberId, val => val.member);
        
        var childNum = popSettings.Size - newMembers.Count;
        var childTasks = new ConcurrentBag<IGenome>();
        Parallel.For(0, childNum, _ =>
        {
            var (parentOneId, parentTwoId) = ParentSelector.Select(popSettings.InbreedRate, Species);
            var parentOne = Members[parentOneId];
            var parentTwo = Members[parentTwoId];
        
            var newGenomeTask = parentOne.Fitness > parentTwo.Fitness 
                ? parentOne.Model.Crossover(parentTwo.Model, envSettings, popSettings.CrossoverRate) 
                : parentTwo.Model.Crossover(parentOne.Model, envSettings, popSettings.CrossoverRate);
            
            childTasks.Add(newGenomeTask);
        });

        foreach (var child in childTasks)
        {
            newMembers[Guid.NewGuid()] = new Member { Fitness = 0, Model = child };
        }

        var ra = RandomGenerator.RandomGenerator.Next;
        Species = Species.Select(niche => niche.Reset(ra)).ToList();

        return new Generation
        {
            Members = newMembers,
            Species = Species,
            MascotMembers = Species
                .Select(spec => (spec.Mascot, Members[spec.Mascot]))
                .ToDictionary(key => key.Mascot, val => val.Item2)
        };
    }

    public void CleanPopulation(double pct)
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
}