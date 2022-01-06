
using Radiate.Domain.Records;

namespace Radiate.Data.Utils;

public class ProgressBar
{
    private readonly int _maxVal;
    private readonly DateTime _startTime;
    
    private DateTime _previousTime;
    private int _tick = 0;
    private int _row;
    private int _col;

    public ProgressBar(int maxVal)
    {
        _maxVal = maxVal;
        _startTime = DateTime.Now;
        _previousTime = DateTime.Now;
    }

    public void Tick(Epoch epoch)
    {
        if (_tick == 0)
        {
            _row = Console.CursorTop;
            _col = Console.CursorLeft;
        }
        
        _tick++;

        var timeSince = DateTime.Now - _startTime;
        var iterationTime = DateTime.Now - _previousTime;
        
        var percent = (float)_tick / _maxVal;
        
        var labels = $"{"Epoch",-7} | {"Total Time",-15} | {"Iter Time",-15} | {"Class Acc",-15} | {"Category Acc",-15} | {"Reg Acc",-15} | {"Loss",-15}";
        var display = $"{_tick,-7} | {timeSince,-15:hh\\:mm\\:ss} | {iterationTime,-15:mm\\:ss\\:fff}" +
            $" | {epoch.ClassificationAccuracy,-15:P2} | {epoch.CategoricalAccuracy,-15:P2} | {epoch.RegressionAccuracy,-15:P2} | {epoch.Loss,-15}";

        var barLength = display.Length - 13;
        var tickPct = (Math.Round((float)_tick / _maxVal * 100) / 100) * barLength;
        var bar = string.Join("", Enumerable.Range(0, barLength).Select(val => val <= tickPct? "#" : " "));

        var pctBar = $"{percent,-7:P2} | {bar}";
        var seperator = string.Join("-", Enumerable.Range(0, pctBar.Length).Select(_ => ""));
        var baseDisplay = $"{"",-7} |";
        
        var toWrite = $"\n\t{labels}\n\t{seperator}\n\t{display}\n\t{seperator}\n\t{pctBar}\n\t{seperator}\n\t{baseDisplay}\n";
        
        Console.SetCursorPosition(_col, _row);
        Console.Write($"{toWrite}");
        
        _previousTime = DateTime.Now;
    }

    public void Tick(string displayString)
    {
        Console.Write($"\r{displayString}");
        
        var backup = new string('\b', displayString.Length);
        Console.Write(backup);
    }

}