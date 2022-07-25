using Radiate.Callbacks.Interfaces;
using Radiate.Optimizers;
using Radiate.Tensors;
using Radiate.Tensors.Enums;

namespace Radiate.Callbacks;

public class PredictionCsvWriterCallback : ITrainingCompletedCallback
{
    public async Task CompleteTraining(Optimizer optimizer, TensorTrainSet pair)
    {
        var csv = "";

        var training = pair.InputsToArrayRow(TrainTest.Train);
        var testing = pair.InputsToArrayRow(TrainTest.Test);
        foreach (var (feature, target) in training)
        {
            var output = optimizer.ProcessedPredict(feature);
            csv += $"{output.Confidence}, {target.Max()}\n";
        }
        
        foreach (var (feature, target) in testing)
        {
            var output = optimizer.ProcessedPredict(feature);
            csv += $"{output.Confidence}, {target.Max()}\n";
        }
        
        var directory = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (directory != null && !directory.GetDirectories("Saves").Any())
        {
            directory = directory.Parent;
        }
        
        var filePath = Path.Combine(directory.FullName, "Saves", $"training_results.csv");

        await File.WriteAllTextAsync(filePath, csv);
    }
}