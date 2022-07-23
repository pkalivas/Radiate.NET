using Radiate.Callbacks.Interfaces;
using Radiate.Optimizers.Evolution;

namespace Radiate.Callbacks;

public class GenerationCallback : IGenerationEvolvedCallback
{
    private int _row;
    private int _col;
    private bool _init = false;
    private int _maxEpoch;

    // public GenerationCallback(int maxEpoch)
    // {
    //     _maxEpoch = maxEpoch;
    // }
    
    public void GenerationEvolved(Generation generation)
    {
        if (!_init)
        {
            _row = Console.CursorTop;
            _col = Console.CursorLeft;
            _init = true;
        }
        
        var report = generation.GetReport();

        var top = $"{"Generation",-15} {report.GenerationNum}\n" +
                  $"{"Member Count",-15} {report.NumMembers}\n" +
                  $"{"Niche Count",-15} {report.NumNiche}\n" +
                  $"{"Fitness",-15} {report.TopFitness}\n" +
                  $"{"Stagnation",-15} {report.StagnationCount}\n" +
                  $"{"Distance",-15} {report.Distance}\n";

        var niches = $"{"Innovation",-10} {"Age",-6} {"Fitness",-10} {"Members",-10}\n";
        var separator = string.Join("-", Enumerable.Range(0, niches.Length).Select(_ => ""));
        niches += separator + "\n";
        foreach (var niche in report.NicheReports)
        {
            niches += $" {niche.Innovation,-10}" +
                      $" {niche.Age,-6}" +
                      $" {Math.Round(niche.AdjustedFitness, 7),-10}" +
                      $" {niche.NumMembers,-10}\n";
        }

        var toWrite = top + niches;
        
        Console.Write($"{toWrite}\n");
        // Console.SetCursorPosition(_col, _row);
    }
}