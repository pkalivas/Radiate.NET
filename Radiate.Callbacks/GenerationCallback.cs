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
    
    public void GenerationEvolved(int index, GenerationReport report)
    {
        if (!_init)
        {
            _row = Console.CursorTop;
            _col = Console.CursorLeft;
            _init = true;
        }

        var top = $"{"Generation",-15} {index}\n" +
                  $"{"GenomeFitnessPair Count",-15} {report.NumMembers}\n" +
                  $"{"Species Count",-15} {report.SpeciesReport.NicheReports.Count}\n" +
                  $"{"Fitness",-15} {report.TopFitness}\n" +
                  $"{"Distance",-15} {report.SpeciesReport.Distance}\n";

        var niches = $"{"Innovation",-10} {"Age",-6} {"Stagnation",-10} {"Members",-10} {"Fitness",-10} {"Min Fit",-10} {"Max Fit",-10}\n";
        var separator = string.Join("-", Enumerable.Range(0, niches.Length).Select(_ => ""));
        niches += separator + "\n";
        foreach (var niche in report.SpeciesReport.NicheReports)
        {
            niches += $" {niche.Innovation,-10}" +
                      $" {niche.Age,-6}" +
                      $" {niche.Stagnation,-6}" +
                      $" {niche.NumMembers,-10}" +
                      $" {Math.Round(niche.AdjustedFitness, 7),-10}" +
                      $" {Math.Round(niche.MinFitness, 7),-10}" +
                      $" {Math.Round(niche.MaxFitness, 7),-10}\n";
        }

        var toWrite = top + niches;
        
        Console.Write($"\r{toWrite}\n");
        // Console.SetCursorPosition(_col, _row);
    }
}