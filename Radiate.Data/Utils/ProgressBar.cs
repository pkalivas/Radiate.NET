﻿
namespace Radiate.Data.Utils;

public class ProgressBar
{
    private readonly int _maxVal;
    private readonly DateTime _startTime;
    private readonly int _barSize;

    private DateTime _previousTime;
    private int _tick = 0;

    public ProgressBar(int maxVal, int barSize = 20)
    {
        _maxVal = maxVal;
        _startTime = DateTime.Now;
        _previousTime = DateTime.Now;
        _barSize = barSize;
    }

    public void Tick(string displayString)
    {
        _tick++;
        
        if (_tick == 1)
        {
            Console.WriteLine("\n");
        }
        
        var percent = Math.Round((float)_tick / _maxVal * 100);
        var tickPct = ((float)percent / 100) * _barSize;
        var bar = string.Join("", Enumerable.Range(0, _barSize).Select(val => val <= tickPct? "#" : " "));
        var timeSince = DateTime.Now - _startTime;
        var iterationTime = DateTime.Now - _previousTime;

        var iterTime = GetIterTime(iterationTime);
        
        var display = $"[{_tick}] " +
            $"{timeSince.Hours}:{timeSince.Minutes}:{timeSince.Seconds}:{timeSince.Milliseconds} " +
            $"{iterTime}/iter {percent}% [{bar}] - "+
            $"{displayString}";
        
        Console.Write($"\r{display}");
        
        _previousTime = DateTime.Now;
        
        var backup = new string('\b', display.Length);
        Console.Write(backup);

        if (_tick == _maxVal)
        {
            Console.WriteLine();
        }
    }

    private string GetIterTime(TimeSpan iterTime) => (iterTime.Minutes, iterTime.Seconds, iterTime.Milliseconds) switch
    {
        (> 0, _, _) => $"{iterTime.Minutes}:{iterTime.Seconds}:{iterTime.Milliseconds}m",
        (<= 0, > 0, _) => $"{iterTime.Seconds}:{iterTime.Milliseconds}s",
        (<= 0, <= 0, > 0) => $"{iterTime.Milliseconds}ms"
    };
}