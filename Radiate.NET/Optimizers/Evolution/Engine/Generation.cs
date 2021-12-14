using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.NET.Optimizers.Evolution.Engine.Delegates;

namespace Radiate.NET.Optimizers.Evolution.Engine
{
    public class Generation 
    {
        public Dictionary<Guid, Member<Genome>> Members { get; set; }
        public Dictionary<Guid, Member<Genome>> MascotMembers { get; set; }
        public List<Niche> Species { get; set; }


        public Generation()
        {
            Members = new Dictionary<Guid, Member<Genome>>();
            MascotMembers = new Dictionary<Guid, Member<Genome>>();
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


        public async Task<Generation> CreateNextGeneration(
            PopulationSettings popSettings,
            EvolutionEnvironment envSettings, 
            GetSurvivors survivorPicker, 
            GetParents parentPicker)
        {
            var newMembers = survivorPicker(Members, Species)
                .Select(pair =>
                {
                    var model = pair.member.Model;
                    model.ResetGenome();
                    return (pair.memberId, member: new Member<Genome> { Fitness = 0, Model = model });
                })
                .ToDictionary(key => key.memberId, val => val.member);

            foreach (var batch in BatchSizes(Members.Keys.Count - newMembers.Count))
            {
                var childTasks = batch.Select(_ => Task.Run(async () =>
                {
                    var (parentOneId, parentTwoId) = parentPicker(popSettings.InbreedRate, Species);
                    var parentOne = Members[parentOneId];
                    var parentTwo = Members[parentTwoId];

                    return parentOne.Fitness > parentTwo.Fitness 
                        ? await parentOne.Model.Crossover(parentTwo.Model, envSettings, popSettings.CrossoverRate) 
                        : await parentTwo.Model.Crossover(parentOne.Model, envSettings, popSettings.CrossoverRate);
                }));

                foreach (var newMember in await Task.WhenAll(childTasks))
                {
                    newMembers[Guid.NewGuid()] = new Member<Genome> {Fitness = 0, Model = newMember};
                }
            }

            Species = Species.Select(niche => niche.Reset()).ToList();

            return new Generation
            {
                Members = newMembers,
                Species = Species,
                MascotMembers = Species
                    .Select(spec => (spec.Mascot, Members[spec.Mascot]))
                    .ToDictionary(key => key.Mascot, val => val.Item2)
            };
        }


        public async Task Optimize(Func<Genome, double> problem)
        {
            foreach (var batch in BatchMembers(Members.Keys.ToList()))
            {
                var fitnessTasks = batch.Select(member => Task.Run(() => (
                    Id: member,
                    Fitness: problem(Members[member].Model)
                )));

                foreach (var (id, fitness) in (await Task.WhenAll(fitnessTasks)))
                {
                    Members[id].Fitness = fitness;
                };
            }

        }


        public void CleanPopulation(double pct)
        {
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
            }
        }


        public Member<Genome> GetBestMember() => 
            Members.Values
                .Aggregate(Members.Values.First(), (best, current) => current.Fitness > best.Fitness ? current : best);
        


        private static IEnumerable<List<Guid>> BatchMembers(List<Guid> memberIds)
        {
            var batchSize = Environment.ProcessorCount;
            var batchCounter = 0;

            while (batchCounter < memberIds.Count)
            {
                yield return memberIds.Skip(batchCounter).Take(batchSize).ToList();
                batchCounter += batchSize;
            }
        }

        private static IEnumerable<IEnumerable<int>> BatchSizes(int size)
        {
            var batchSize = Environment.ProcessorCount;
            var batchCounter = 0;

            while (batchCounter < size)
            {
                yield return batchCounter + batchSize > size 
                    ? Enumerable.Range(0, size - batchCounter)
                    : Enumerable.Range(0, batchSize);
                batchCounter += batchSize;
            }
        }


    }
}