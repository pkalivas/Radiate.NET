using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Optimizers.Evolution.Engine;
using Radiate.Optimizers.Evolution.Engine.Delegates;
using Radiate.Optimizers.Evolution.Engine.ParentalCriteria;
using Radiate.Optimizers.Evolution.Engine.SurvivorCriteria;

namespace Radiate.Optimizers.Evolution
{
    public class Population : IPopulation
    {
        private PopulationSettings Settings { get; set; }
        private Generation CurrentGeneration { get; set; }
        private EvolutionEnvironment EvolutionEnvironment { get; set; }
        private Func<Genome, double> Solver { get; set; }
        private GetSurvivors SurvivorPicker { get; set; }
        private GetParents ParentPicker { get; set; }
        private int GenerationsUnchanged { get; set; }
        private double PreviousFitness { get; set; }
        private const double Tolerance = 0.0000001;

        public Population(Func<Genome, double> fitnessFunction) : this(new PopulationSettings(), new BaseEvolutionEnvironment(), fitnessFunction) { } 
        
        public Population(PopulationSettings settings, EvolutionEnvironment env, Func<Genome, double> solve) : this(settings, env, solve, new Fittest().Pick, new BiasedRandom().Pick) { }

        public Population(PopulationSettings settings, EvolutionEnvironment env, Func<Genome, double> solve, GetSurvivors survivorPicker, GetParents parentPicker)
        {
            Settings = settings;
            EvolutionEnvironment = env;
            Solver = solve;
            SurvivorPicker = survivorPicker;
            ParentPicker = parentPicker;
            
        }
        
        public async Task<Member<Genome>> Evolve(Genome genome, Run runFunction)
        {
            PopulateClone(genome);
            
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

        private void PopulateClone(Genome baseGenome)
        {
            CurrentGeneration = new Generation
            {
                Members = Enumerable.Range(0, Settings.Size)
                    .Select(_ => (
                        Id: Guid.NewGuid(), 
                        Member: new Member<Genome>
                        {
                            Fitness = 0,
                            Model = baseGenome.CloneGenome()
                        }
                     ))
                    .ToDictionary(key => key.Id, val => val.Member),
                Species = new List<Niche>()
            };
        }
        
        private async Task<Member<Genome>> EvolveGeneration()
        {
            await CurrentGeneration.Optimize(Solver);

            var topMember = CurrentGeneration.GetBestMember();

            await CurrentGeneration.Speciate(Settings.SpeciesDistance, EvolutionEnvironment);
            
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