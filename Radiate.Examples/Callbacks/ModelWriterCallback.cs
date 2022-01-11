using System.IO;
using Newtonsoft.Json;
using Radiate.Domain.Callbacks.Interfaces;
using Radiate.Domain.Loss;
using Radiate.Domain.Records;
using Radiate.Domain.Tensors;
using Radiate.Optimizers;
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

    public async Task CompleteTraining<T>(Optimizer<T> optimizer, List<Epoch> epochs, TensorTrainSet _) where T : class
    {
        var directory = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (directory != null && !directory.GetDirectories("Saves").Any())
        {
            directory = directory.Parent;
        }

        var wrapped = optimizer.Save();
        var modelType = wrapped.ModelWrap.ModelType.GetType().ToString().Split(".").Last();
        modelType += string.IsNullOrEmpty(_ext) ? "" : $"_{_ext}";

        var filePath = Path.Combine(directory.FullName, "Saves", $"{modelType}.json");

        var content = JsonConvert.SerializeObject(wrapped);
        await File.WriteAllTextAsync(filePath, content);
    }
}