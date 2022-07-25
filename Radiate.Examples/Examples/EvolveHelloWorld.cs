using Radiate.Callbacks;
using Radiate.Callbacks.Interfaces;
using Radiate.Optimizers;
using Radiate.Optimizers.Evolution;
using Radiate.Optimizers.Evolution.Info;
using Radiate.Optimizers.Evolution.Interfaces;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Examples.Examples;

public class EvolveHelloWorld : IExample
{

    private static readonly char[] Alphabet = new char[29] {
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
        var target = new[] { 'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!'};

        var worlds = Enumerable.Range(0, populationSize).Select(_ => new HelloWorld()).ToList();

        var info = new PopulationInfo<HelloWorld>()
            .AddSettings(settings =>
            {
                settings.DynamicDistance = true;
                settings.SpeciesTarget = 5;
                settings.SpeciesDistance = .5;
                settings.InbreedRate = .001;
                settings.CrossoverRate = .5;
                settings.StagnationLimit = 15;
            })
            .AddFitnessFunction(member => member.Chars.Zip(target)
                    .Sum(points => points.First == points.Second ? 1.0f : 0.0f));

        var population = new Population<HelloWorld>(info, worlds);
        var optimizer = new Optimizer(population, null, new List<ITrainingCallback>
        {
            new GenerationCallback()
        });
        var model = await optimizer.Train<HelloWorld>(epoch => epoch.Index == evolutionEpochs || epoch.Fitness == 12);
        
        Console.WriteLine($"\nFinal Result: {model.Print()}");
    }
    
    private class HelloWorld : IGenome
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
    
        public string Print() => string.Join("", Chars);
        

        T IGenome.Crossover<T, TE>(T other, TE environment, double crossoverRate)
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

        public double Distance<T>(T other, DistanceControl _)
        {
            var secondParent = other as HelloWorld;
            var total = 0.0;
            foreach (var (pOne, pTwo) in Chars.Zip(secondParent.Chars))
            {
                total += pOne == pTwo ? 1 : 0;
            }

            return Chars.Length / total;
        }

        T IGenome.CloneGenome<T>()
        {
            return new HelloWorld { Chars = Chars.Select(c => c).ToArray() } as T;
        }

        public void ResetGenome() { }
        
        public Prediction Predict(Tensor input)
        {
            throw new NotImplementedException();
        }
    }
}