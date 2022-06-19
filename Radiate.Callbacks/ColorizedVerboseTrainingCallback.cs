using Radiate.Callbacks.Interfaces;
using Radiate.Callbacks.Models;
using Radiate.Optimizers;
using Radiate.Records;
using Radiate.Tensors;
using Spectre.Console;

namespace Radiate.Callbacks;

public class ColorizedVerboseTrainingCallback : IEpochStartedCallback, 
    IEpochCompletedCallback, 
    IBatchCompletedCallback, 
    ITrainingCompletedCallback
{
    private const int BarLength = 30;
    
    private readonly bool _updateOnBatch;
    private readonly TrainingTracker _tracker;
    private readonly Task _displayTableTask;
    private readonly SemaphoreSlim _signal = new(0);

    public ColorizedVerboseTrainingCallback(TensorTrainSet trainSet, int maxEpoch = 1, bool updateOnBatch = true)
    {
        _tracker = new TrainingTracker(maxEpoch, trainSet.TrainingInputs.Count);
        _updateOnBatch = updateOnBatch;
        _displayTableTask = BuildTableDisplay();
    }
    
    public void EpochStarted()
    {
        _tracker.StartEpoch();
    }

    public void EpochCompleted(Epoch epoch)
    {
        _tracker.NextEpoch(epoch);
        _signal.Release();
    }

    public void BatchCompleted(List<Step> steps)
    {
        _tracker.NextBatch(steps);

        if (_updateOnBatch)
        {
            _signal.Release();
        }
    }

    public async Task CompleteTraining(Optimizer optimizer, TensorTrainSet tensorSet)
    {
        _signal.Release();
        await _displayTableTask;
    }

    private Task BuildTableDisplay()
    {
        var table = GetTable();

        return AnsiConsole.Live(table)
            .StartAsync(async ctx =>
            {
                while (_tracker.PreviousEpoch.Index != _tracker.MaxEpochs)
                {
                    ctx.UpdateTarget(GetTable());
                    ctx.Refresh();
                    await _signal.WaitAsync();
                }
                
                ctx.UpdateTarget(GetTable());
                ctx.Refresh();
            });
    }

    private Table GetTable()
    {
        var trainingTable = new Table()
            .SimpleBorder()
            .Expand()
            .BorderColor(Color.Grey)
            .AddColumn("Class Acc")
            .AddColumn("Category Acc")
            .AddColumn("Reg Acc")
            .AddColumn("Loss")
            .AddRow($"{_tracker.PreviousEpoch.ClassificationAccuracy:P2}",
                    $"{_tracker.PreviousEpoch.CategoricalAccuracy:P2}",
                    $"{_tracker.PreviousEpoch.RegressionAccuracy:P2}",
                    $"{_tracker.PreviousEpoch.Loss}");
        
        var progressTable = new Table()
                .SimpleBorder()
                .Expand()
                .BorderColor(Color.Grey)
                .AddColumns("Batch", $"{_tracker.BatchCount} of {_tracker.MaxBatches}",
                    $"{TimeSpan.FromMilliseconds(_tracker.AvgStepTime):ffff}ms / step", 
                    $"{_tracker.BatchTime:mm\\:ss\\:ffff}", BuildBatchProgressString())
                .AddRow("Epoch", $"{_tracker.PreviousEpoch.Index} of {_tracker.MaxEpochs}",
                    $"{_tracker.PreviousEpoch.avgStepTime:ffff}ms / step",
                    $"{(_tracker.EpochStart - _tracker.EpochEnd):mm\\:ss\\:ffff}",
                    BuildEpochProgressString());

        return new Table()
            .RoundedBorder()
            .Expand()
            .BorderColor(Color.Yellow)
            .AddColumn($"{DateTime.Now - _tracker.StartTime,-10:hh\\:mm\\:ss}")
            .AddColumn("Training")
            .AddColumn("Progress")
            .AddRow(new Text(""), trainingTable, progressTable);
    }

    private string BuildEpochProgressString() =>
        string.Join("", Enumerable
            .Range(0, BarLength)
            .Select(val => val < _tracker.PctEpochDone * BarLength ? "#" : " "));
    
    private string BuildBatchProgressString() =>
        string.Join("", Enumerable
            .Range(0, BarLength)
            .Select(val => val < _tracker.PctBatchDone * BarLength ? "#" : " "));

}