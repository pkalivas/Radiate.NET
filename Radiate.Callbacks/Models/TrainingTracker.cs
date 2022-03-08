using Radiate.Records;

namespace Radiate.Callbacks.Models;

public class TrainingTracker
{
    public int MaxEpochs { get; set; }
    public int MaxBatches { get; set; }
    public int BatchCount { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime EpochStart { get; set; }
    public DateTime EpochEnd { get; set; }
    public DateTime BatchStart { get; set; }
    public TimeSpan BatchTime { get; set; }
    public Epoch PreviousEpoch { get; set; }
    public List<Step> BatchSteps { get; set; } = new();

    public TrainingTracker(int maxEpochs, int batchCount)
    {
        MaxEpochs = maxEpochs;
        MaxBatches = batchCount;
        BatchCount = 0;
        StartTime = DateTime.Now;
        EpochStart = DateTime.Now;
        EpochEnd = DateTime.Now;
        PreviousEpoch = new(0);
    }
    
    public void StartEpoch()
    {
        EpochEnd = EpochStart;
        EpochStart = DateTime.Now;
        BatchStart = DateTime.Now;
    }

    public void NextEpoch(Epoch epoch)
    {
        PreviousEpoch = epoch;
        BatchCount = 0;
    }

    public void NextBatch(List<Step> steps)
    {
        BatchTime = DateTime.Now - BatchStart;
        BatchStart = DateTime.Now;
        BatchCount++;
        BatchSteps = steps;
    }
    
    public double AvgStepTime => BatchSteps.Sum(step => step.Time.TotalMilliseconds) / BatchSteps.Count == 0
        ? 1.0
        : BatchSteps.Count;

    public double PctBatchDone => BatchCount / (double) MaxBatches;
    public double PctEpochDone => PreviousEpoch.Index / (double)MaxEpochs;
}