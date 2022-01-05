﻿using Radiate.Data.Utils;
using Radiate.Optimizers;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Population;

namespace Radiate.Examples.Examples;

public class EvolveHelloWorld : IExample
{

    private static char[] Alphabet = new char[29] {
        '!', ' ', 'a', 'b', 'c', 'd', 'e',
        'f', 'g', 'h', 'i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'q', 'r', 's', 
        't', 'u', 'v', 'w', 'x', 'y', 'z',
        ' '
    };

    public async Task Run()
    {
        const int evolutionEpochs = 500;
        const int populationSize = 100;
        var progressBar = new ProgressBar(evolutionEpochs);
        var target = new char[12] { 'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!'};

        var worlds = new List<HelloWorld>();
        foreach (var _ in Enumerable.Range(0, populationSize))
        {
            worlds.Add(new HelloWorld());
        }

        var population = new Population<HelloWorld, BaseEvolutionEnvironment>(worlds)
            .Configure(settings =>
            {
                settings.Size = populationSize;
                settings.DynamicDistance = true;
                settings.SpeciesTarget = 5;
                settings.SpeciesDistance = .5;
                settings.InbreedRate = .001;
                settings.CrossoverRate = .5;
                settings.CleanPct = .9;
                settings.StagnationLimit = 15;
            })
            .SetFitnessFunction(member => member.Chars.Zip(target)
                    .Sum(points => points.First == points.Second ? 1.0f : 0.0f));

        var optimizer = new Optimizer<Population<HelloWorld, BaseEvolutionEnvironment>>(population);
        await optimizer.Train(epoch =>
        {
            var displayString = $"Fitness: {epoch.Fitness}";
            progressBar.Tick(displayString);
            return epoch.Index == evolutionEpochs || epoch.Fitness == 12;
        });

        var best = optimizer.Model.Best;
        Console.WriteLine($"\nFinal Result: {best.Print()}");
    }
    
    private class HelloWorld : Genome
    {
        public char[] Chars { get; set; }
    
        public HelloWorld()
        {
            var r = new Random();
            Chars = Enumerable.Range(0, 12).Select(val =>
                {
                    var charIndex = r.Next(0, Alphabet.Length);
                    return Alphabet[charIndex];
                })
                .ToArray();
        }
    
        public string Print() => String.Join("", Chars);
    
        public override async Task<T> Crossover<T, TE>(T other, TE environment, double crossoverRate)        
        {
            var child = new HelloWorld();
            var secondParent = other as HelloWorld;
            var r = new Random();
    
            if (r.NextDouble() < crossoverRate)
            {
                var childAlph = new List<char>();
                foreach (var (pOne, pTwo) in Chars.Zip(secondParent.Chars))
                {
                    childAlph.Add(pOne != pTwo ? pOne : pTwo);
                }
    
                child.Chars = childAlph.ToArray();
            }
            else
            {
                var newData = secondParent.Chars.Select(c => c).ToArray();
                var swapIndex = r.Next(0, newData.Length);
                var newCharIndex = r.Next(0, Alphabet.Length);
                newData[swapIndex] = Alphabet[newCharIndex];
    
                child.Chars = newData;
            }

            return child as T;
        }
    
        public override async Task<double> Distance<T, TE>(T other, TE environment)
        {
            var secondParent = other as HelloWorld;
            var total = 0.0;
            foreach (var (pOne, pTwo) in Chars.Zip(secondParent.Chars))
            {
                total += pOne == pTwo ? 1 : 0;
            }

            return Chars.Length / total;
        }
        
        public override T CloneGenome<T>()
        {
            return new HelloWorld { Chars = Chars.Select(c => c).ToArray() } as T;
        }
    
        public override void ResetGenome() { }
    }
}