using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Radiate.Data.Utils;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Engine;

namespace Radiate.Examples.Examples
{
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
            var evolutionEpochs = 500;
            var helloWorld = new HelloWorld();
            var target = new char[12] { 'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!'};

            var fitnessFunction = new Func<Genome, double>((Genome member) =>
            {
                var casted = member as HelloWorld;
                var total = casted.Chars.Zip(target).Sum(points => points.First == points.Second ? 1.0 : 0.0);
                return total;
            });

            var population = new Population(fitnessFunction);

            var progressBar = new ProgressBar(evolutionEpochs);
            var bestMember = await population.Evolve(helloWorld, (member, epoch) =>
            {
                var world = member.Model as HelloWorld;
                var displayString = $"Genome: {world.ToString()} Fitness: {member.Fitness}";
                progressBar.Tick(displayString);
                return epoch == evolutionEpochs || member.Fitness == 12;
            });

            Console.WriteLine($"\nFinal Result: {(bestMember.Model as HelloWorld).ToString()} - Fitness: {bestMember.Fitness}");
        }


        public class HelloWorld : Genome
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

            public string ToString() => String.Join("", Chars);

            public override Task<Genome> Crossover(Genome other, EvolutionEnvironment environment, double crossoverRate)
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

                return Task.Run(() => child as Genome);
            }

            public override Task<double> Distance(Genome other, EvolutionEnvironment environment)
            {
                var secondParent = other as HelloWorld;
                var total = 0.0;
                foreach (var (pOne, pTwo) in Chars.Zip(secondParent.Chars))
                {
                    total += pOne == pTwo ? 1 : 0;
                }

                return Task.Run(() => Chars.Length / total);
            }

            public override float[] Forward(float[] data)
            {
                throw new System.NotImplementedException();
            }

            public override Genome CloneGenome()
            {
                return new HelloWorld { Chars = Chars.Select(c => c).ToArray() };
            }

            public override void ResetGenome() { }
        }
    }
}