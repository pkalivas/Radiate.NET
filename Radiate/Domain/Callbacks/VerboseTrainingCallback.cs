using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;

namespace Radiate.Domain.Callbacks;

public class VerboseTrainingCallback : IEpochCompletedCallback, IBatchCompletedCallback
{
    private readonly int _maxEpoch;
    private readonly DateTime _startTime;
    
    private DateTime _previousTime;
    private int _row;
    private int _col;
    private Epoch _previousEpoch;

    public VerboseTrainingCallback(int maxEpoch)
    {
        _maxEpoch = maxEpoch;
        _previousEpoch = new Epoch(0);
    }

    public void EpochCompleted(Epoch epoch)
    {
        if (epoch.Index == 0)
        {
            _row = Console.CursorTop;
            _col = Console.CursorLeft;
        }
        
        var timeSince = DateTime.Now - _startTime;
        var iterationTime = DateTime.Now - _previousTime;
        
        var percent = (float)epoch.Index / _maxEpoch;
        
        var labels = $"{"Epoch",-7} |" +
                     $" {"Total Time",-15} |" +
                     $" {"Iter Time",-15} |" +
                     $" {"Class Acc",-15} |" +
                     $" {"Category Acc",-15} |" +
                     $" {"Reg Acc",-15} |" +
                     $" {"Loss",-15}";
        
        var display = $"{epoch.Index,-7} |" +
                      $" {timeSince,-15:hh\\:mm\\:ss} |" +
                      $" {iterationTime,-15:mm\\:ss\\:fff} |"+
                      $" {epoch.ClassificationAccuracy,-15:P2} |" +
                      $" {epoch.CategoricalAccuracy,-15:P2} |" +
                      $" {epoch.RegressionAccuracy,-15:P2} |" +
                      $" {epoch.Loss,-15}";

        var barLength = display.Length - 13;
        var tickPct = (Math.Round((float)epoch.Index / _maxEpoch * 100) / 100) * barLength;
        var bar = string.Join("", Enumerable.Range(0, barLength).Select(val => val <= tickPct? "#" : " "));

        var pctBar = $"{percent,-7:P2} | {bar}";
        var separator = string.Join("-", Enumerable.Range(0, pctBar.Length).Select(_ => ""));
        var baseDisplay = $"{"",-7} |";
        
        var toWrite = $"\n\t{labels}\n\t{separator}\n\t{display}\n\t{separator}\n\t{pctBar}\n\t{separator}\n\t{baseDisplay}\n";
        
        Console.SetCursorPosition(_col, _row);
        Console.Write($"{toWrite}");
        
        _previousTime = DateTime.Now;
        _previousEpoch = epoch;
    }
    
        
    public void BatchCompleted(List<Prediction> predictions, List<Tensor> targets)
    {
        
    }

}