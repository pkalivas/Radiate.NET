using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.NET.Models.Neat.Enums; 
using Radiate.NET.Models.Neat.Layers;
using Radiate.NET.Engine;
using Radiate.NET.Engine.Enums;
using Radiate.NET.Engine.ParentalCriteria;
using Radiate.NET.Engine.SurvivorCriteria;
using Radiate.NET.Models.Neat;

namespace Radiate.NET
{
    class Program
    {
        static void Main(string[] args)
        {
            Run().ConfigureAwait(true).GetAwaiter().GetResult();
        }


        private static async Task Run()
        {
            var inputs = new List<List<float>>();
            inputs.Add(new List<float> { 0, 0 });
            inputs.Add(new List<float> { 1, 1 });
            inputs.Add(new List<float> { 1, 0 });
            inputs.Add(new List<float> { 0, 1 });

            var answers = new List<List<double>>();
            answers.Add(new List<double> { 0 });
            answers.Add(new List<double> { 0 });
            answers.Add(new List<double> { 1 });
            answers.Add(new List<double> { 1 });

            var neat = new Neat()
                .AddLayer(new Dense(2, 1, ActivationFunction.Sigmoid, LayerType.DensePool));

            var populationMembers = Enumerable.Range(0, 100)
                .Select(_ => new Neat().AddLayer(new Dense(2, 1, ActivationFunction.Sigmoid, LayerType.DensePool)))
                .ToList();

            var startTime = DateTime.Now;

            var best = await new Population<Neat, NeatEnvironment>()
                .Configure(settings =>
                { 
                    settings.Size = 250;
                    settings.DynamicDistance = true;
                    settings.SpeciesTarget = 2;
                    settings.SpeciesDistance = .5;
                    settings.InbreedRate = .001;
                    settings.CrossoverRate = .5;
                    settings.CleanPct = .9;
                    settings.StagnationLimit = 15;
                })
                .SetParentPicker(ParentPickerResolver.Get<Neat>(ParentPicker.BiasedRandom))
                .SetSurvivorPicker(SurvivorPickerResolver.Get<Neat>(SurvivorPicker.Fittest))
                .SetEnvironment(new NeatEnvironment
                {
                    ReactivateRate = .2f,
                    WeightMutateRate = .8f,
                    RecurrentNeuronRate = .0f,
                    NewEdgeRate = .08f,
                    NewNodeRate = .08f,
                    EditWeights = .1f,
                    WeightPerturb = 1.5f,
                    ActivationFunctions = new List<ActivationFunction>
                    {
                        ActivationFunction.Sigmoid,
                        ActivationFunction.Relu
                    }
                })
                .SetSolver(member =>
                {
                    var total = 0.0;
                    foreach (var points in inputs.Zip(answers))
                    {
                        var output = member.Forward(points.First);
                        total += Math.Pow((output[0] - points.Second[0]), 2);
                    }

                    return 4.0 - total;
                })
                .PopulateClone(neat)
                .Train((member, epoch) =>
                {
                    Console.WriteLine($"{member.Fitness} - {epoch}");
                    if (epoch == 250)
                    {
                        return true;
                    }

                    return false;
                });

            var member = best.Model;
            member.ResetGenome();
            
            foreach (var (point, idx) in inputs.Select((val, idx) => (val, idx)))
            {
                var output = member.Forward(point);
                Console.WriteLine($"Input ({point[0]} {point[1]}) output ({output[0]} answer ({answers[idx][0]})");
            }


            var endTime = DateTime.Now - startTime;
            Console.WriteLine($"{endTime.Milliseconds}");

            //var options = new JsonSerializerOptions
            //{
            //    WriteIndented = true,
            //};
            //var model = System.Text.Json.JsonSerializer.Serialize(member.Wrap(), options);
            //await File.WriteAllTextAsync(@"C:\Users\Peter\Desktop\Radiate.NET\neat.json", model);
        }

    }
}
