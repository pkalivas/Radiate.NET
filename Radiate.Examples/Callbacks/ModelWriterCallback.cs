using System.IO;
using Newtonsoft.Json;
using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Unsupervised;

namespace Radiate.Examples.Callbacks;

public class ModelWriterCallback : ITrainingCompletedCallback
{
    private readonly string _ext;

    public ModelWriterCallback(string ext = "")
    {
        _ext = ext;
    }
    
    public async Task CompleteTraining<T>(T model, List<Epoch> epochs, LossFunction _)
    {
        var directory = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (directory != null && !directory.GetDirectories("Saves").Any())
        {
            directory = directory.Parent;
        }

        var modelType = model.GetType().ToString().Split(".").Last();
        modelType += string.IsNullOrEmpty(_ext) ? "" : $"_{_ext}";
        
        var filePath = Path.Combine(directory.FullName, "Saves", $"{modelType}.json");

        var content = model switch
        {
            ISupervised supervised => JsonConvert.SerializeObject(supervised.Save()),
            IUnsupervised unsupervised => JsonConvert.SerializeObject(unsupervised.Save()),
            _ => throw new Exception("Cannot serialize model.")
        };

        await File.WriteAllTextAsync(filePath, content);
    }
}