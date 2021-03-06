using Radiate.Callbacks.Interfaces;
using Radiate.Optimizers;
using Radiate.Records;
using Radiate.Tensors;

namespace Radiate.Callbacks;

public class VerboseTrainingCallback : IEpochStartedCallback, 
    IEpochCompletedCallback, 
    IBatchCompletedCallback, 
    ITrainingCompletedCallback
{
    private readonly int _maxEpoch;
    private readonly int _numFeatures;
    private readonly DateTime _startTime;

    private DateTime _epochStart;
    private DateTime _epochEnd;
    private DateTime _previousTime;
    private int _row;
    private int _col;
    private Epoch _previousEpoch;
    private int _batchCount;
    private bool _init = false;
    private readonly bool _updateOnBatch;
    

    public VerboseTrainingCallback(TensorTrainSet trainSet, int maxEpoch = 1, bool updateOnBatch = true)
    {
        _maxEpoch = maxEpoch;
        _numFeatures = trainSet.TrainingInputs.Count;
        _startTime = DateTime.Now;
        _previousTime = DateTime.Now;
        _epochStart = DateTime.Now;
        _epochEnd = DateTime.Now;
        _previousEpoch = new Epoch(0);
        _batchCount = 0;
        _updateOnBatch = updateOnBatch;
    }

    public void EpochStarted()
    {
        _epochEnd = _epochStart;
        _epochStart = DateTime.Now;
    }

    public void EpochCompleted(Epoch epoch)
    {
        _previousEpoch = epoch;
        _batchCount = 0;
        Write();
    }
    
        
    public void BatchCompleted(List<Step> steps)
    {
        _batchCount++;

        if (_updateOnBatch)
        {
            Write();
        }
    }

    private void Write()
    {
        if (!_init)
        {
            _row = Console.CursorTop;
            _col = Console.CursorLeft;
            _init = true;
        }
        
        var iterationTime = DateTime.Now - _previousTime;
        
        var labels = $"{"Training",-10} |" +
                     $" {"Class Acc",-15} |" +
                     $" {"Category Acc",-15} |" +
                     $" {"Reg Acc",-15} |" +
                     $" {"Loss",-15}";
        
        var display = $"{DateTime.Now - _startTime,-10:hh\\:mm\\:ss} |" +
                      $" {_previousEpoch.ClassificationAccuracy,-15:P2} |" +
                      $" {_previousEpoch.CategoricalAccuracy,-15:P2} |" +
                      $" {_previousEpoch.RegressionAccuracy,-15:P2} |" +
                      $" {_previousEpoch.Loss,-15}";

        var trainPct = _previousEpoch.Index / (float)_maxEpoch;
        var batchPct = _batchCount / (float)_numFeatures;

        var pctTrain = $"{"Epoch",-10} | {_epochEnd - _epochStart,-15:hh\\:mm\\:ss} | {$"{_previousEpoch.Index}/{_maxEpoch}",-15} |";
        var pctBatch = $"{"Batch",-10} | {iterationTime,-15:mm\\:ss\\:fff} | {$"{_batchCount}/{_numFeatures}",-15} |";

        var barLength = display.Length - pctTrain.Length - 9;
        var trainBar = string.Join("", Enumerable.Range(0, barLength).Select(val => val <= trainPct * barLength? "#" : " "));
        var batchBar = string.Join("", Enumerable.Range(0, barLength).Select(val => val <= batchPct * barLength? "#" : " "));

        pctTrain += $" {trainPct:P2} {trainBar}";
        pctBatch += $" {batchPct:P2} {batchBar}";
        
        var separator = string.Join("-", Enumerable.Range(0, display.Length).Select(_ => ""));

        var toWrite = $"\n\t{labels}" +
                      $"\n\t{separator}" +
                      $"\n\t{display}" +
                      $"\n\t{separator}" +
                      $"\n\t{pctTrain}" +
                      $"\n\t{separator}" +
                      $"\n\t{pctBatch}" +
                      $"\n\t{separator}\n";
        
        

        if (_previousEpoch.Index != _maxEpoch)
        {
            Console.Write($"\r{toWrite}");
            Console.SetCursorPosition(_col, _row);
        }
        else
        {
            Console.SetCursorPosition(_col, _row);
            Console.Write($"\r{toWrite}");
        }

        _previousTime = DateTime.Now;
    }

    public Task CompleteTraining(Optimizer optimizer, TensorTrainSet tensorSet)
    {
        var validation = optimizer.ValidationScores();
        
        var labels = $"{"Validation",-7} |" +
                     $" {"Class Acc",-15} |" +
                     $" {"Category Acc",-15} |" +
                     $" {"Reg Acc",-15} |" +
                     $" {"Loss",-15}";

        var testDisplay = $"{"",-10} |" +
                          $" {validation.ClassificationAccuracy,-15:P2} |" +
                          $" {validation.CategoricalAccuracy,-15:P2} |" +
                          $" {validation.RegressionAccuracy,-15:P2} |" +
                          $" {validation.Loss,-15}";

        var separator = string.Join("-", Enumerable.Range(0, labels.Length).Select(_ => ""));
        var toWrite = $"\n\t{labels}\n\t{separator}\n\t{testDisplay}";
        
        Console.Write($"{toWrite}");
        
        return Task.CompletedTask;
    }
}