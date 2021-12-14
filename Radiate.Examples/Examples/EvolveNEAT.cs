using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Data;
using Radiate.Domain.Activation;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Engine;
using Radiate.Optimizers.Evolution.Neat;

namespace Radiate.Examples.Examples
{
    public class EvolveNEAT : IExample
    {
        public async Task Run()
        {
            var (inputs, answers) = new XOR().GetDataSet();

            var neat = new Neat(2, 1, Activation.ExpSigmoid);
            var popSettings = new PopulationSettings
            {
                Size = 100,
                DynamicDistance = true,
                SpeciesTarget = 5,
                SpeciesDistance = .5,
                InbreedRate = .001,
                CrossoverRate = .5,
                CleanPct = .9,
                StagnationLimit = 15,
            };

            var neatEnvironment = new NeatEnvironment
            {
                ReactivateRate = .2f,
                WeightMutateRate = .8f,
                NewEdgeRate = .14f,
                NewNodeRate = .14f,
                EditWeights = .1f,
                WeightPerturb = 1.5f,
                ActivationFunctions = new List<Activation>
                {
                    Activation.ExpSigmoid,
                    Activation.ReLU
                }
            };

            var fitnessFunction = new Func<Genome, double>((Genome member) =>
            {
                var total = 0.0;
                foreach (var points in inputs.Zip(answers))
                {
                    var output = member.Forward(points.First);
                    total += Math.Pow((output[0] - points.Second[0]), 2);
                }

                return 4.0 - total;
            });
            
            var population = new Population(popSettings, neatEnvironment, fitnessFunction);

            var best = await population.Evolve(neat, ((member, epoch) => epoch == 500));

            var member = best.Model;
            member.ResetGenome();
            
            foreach (var (point, idx) in inputs.Select((val, idx) => (val, idx)))
            {
                var output = member.Forward(point);
                Console.WriteLine($"Input ({point[0]} {point[1]}) output ({output[0]} answer ({answers[idx][0]})");
            }
        }
    }
}