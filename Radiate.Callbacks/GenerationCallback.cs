using Radiate.Callbacks.Interfaces;
using Radiate.Optimizers.Evolution;
using Radiate.Records;

namespace Radiate.Callbacks;

public class GenerationCallback : IGenerationEvolvedCallback
{
    public void GenerationEvolved(Generation generation, PopulationControl populationControl)
    {
        var report = generation.GetReport();

        var top = $"{"Generation",-10} {report.GenerationNum}\n" +
                  $"{"Member Count",-10} {report.NumMembers}\n" +
                  $"{"Niche Count",-10} {report.NumNiche}\n" +
                  $"{"Fitness",-10} {report.TopFitness}";
        
        Console.WriteLine($"{top}\n");
    }
}