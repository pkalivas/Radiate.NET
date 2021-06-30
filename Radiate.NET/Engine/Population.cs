using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.NET.Engine.Delegates;

namespace Radiate.NET.Engine
{
    public class Population<T, E>
        where T: Genome
        where E: EvolutionEnvironment
    {
        private PopulationSettings Settings { get; set; }
        private Generation<T, E> CurrentGeneration { get; set; }
        private EvolutionEnvironment EvolutionEnvironment { get; set; }
        private Solve<T> Solver { get; set; }
        private GetSurvivors<T> SurvivorPicker { get; set; }
        private GetParents<T> ParentPicker { get; set; }
        private int GenerationsUnchanged { get; set; }
        private double PreviousFitness { get; set; }
        private const double Tolerance = 0.0000001;


        public Population()
        {
            Settings = new PopulationSettings();
        }

        public Population<T, E> Configure(Action<PopulationSettings> settings)
        {
            settings.Invoke(Settings);

            return this;
        }

        public Population<T, E> SetSurvivorPicker(GetSurvivors<T> survivors)
        {
            SurvivorPicker = survivors;

            return this;
        }

        public Population<T, E> SetParentPicker(GetParents<T> parents)
        {
            ParentPicker = parents;

            return this;
        }

        public Population<T, E> SetEnvironment(EvolutionEnvironment environment)
        {
            EvolutionEnvironment = environment;

            return this;
        }


        public Population<T, E> PopulateClone(T baseGenome)
        {
            CurrentGeneration = new Generation<T, E>
            {
                Members = Enumerable.Range(0, Settings.Size)
                    .Select(_ => (
                        Id: Guid.NewGuid(), 
                        Member: new Member<T>
                        {
                            Fitness = 0,
                            Model = baseGenome.CloneGenome<T>()
                        }
                     ))
                    .ToDictionary(key => key.Id, val => val.Member),
                Species = new List<Niche>()
            };

            return this;
        }


        public Population<T, E> Populate(List<T> members)
        {
            CurrentGeneration = new Generation<T, E>
            {
                Members = members
                    .Select(member => (
                        Id: Guid.NewGuid(),
                        Member: new Member<T>
                        {
                            Fitness = 0,
                            Model = member
                        }
                    ))
                    .ToDictionary(key => key.Id, val => val.Member),
                Species = new List<Niche>()
            };

            return this;
        }

        public Population<T, E> SetSolver(Solve<T> solver)
        {
            Solver = solver;

            return this;
        }


        public async Task<Member<T>> Train(Run<T> runFunction)
        {
            var epoch = 0;
            while (true)
            {
                var topMember = await EvolveGeneration();

                if (runFunction(topMember, epoch))
                {
                    return topMember;
                }

                epoch++;
            }
        }


        private async Task<Member<T>> EvolveGeneration()
        {
            await CurrentGeneration.Optimize(Solver);

            var topMember = CurrentGeneration.GetBestMember();

            await CurrentGeneration.Speciate(Settings.SpeciesDistance, EvolutionEnvironment);

            foreach (var species in CurrentGeneration.Species)
            {
                //Console.WriteLine($"Species: {species.NicheId} Gens: {species.Age} Members: {species.Members.Count} Fit: {species.TotalAdjustedFitness}");
            }

            //Console.WriteLine();

            if (Settings.DynamicDistance)
            {
                if (CurrentGeneration.Species.Count < Settings.SpeciesTarget)
                {
                    Settings.SpeciesDistance -= .1;
                }
                else if (CurrentGeneration.Species.Count > Settings.SpeciesTarget)
                {
                    Settings.SpeciesDistance += .1;
                }

                if (Settings.SpeciesDistance < .01)
                {
                    Settings.SpeciesDistance = .01;
                }
            }

            if (GenerationsUnchanged >= Settings.StagnationLimit)
            {
                CurrentGeneration.CleanPopulation(Settings.CleanPct);
                GenerationsUnchanged = 0;
            }
            else if (Math.Abs(PreviousFitness - topMember.Fitness) < Tolerance)
            {
                GenerationsUnchanged++;
            }
            else
            {
                GenerationsUnchanged = 0;
            }
            
            PreviousFitness = topMember.Fitness;

            CurrentGeneration = await CurrentGeneration.CreateNextGeneration(Settings, EvolutionEnvironment, SurvivorPicker, ParentPicker);

            return topMember;
        }
        
    }


}