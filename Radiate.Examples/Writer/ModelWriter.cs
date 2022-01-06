using System.IO;
using Newtonsoft.Json;
using Radiate.Optimizers.Supervised;
using Radiate.Optimizers.Unsupervised;

namespace Radiate.Examples.Writer;

public static class ModelWriter
{
    public static async Task Write<T>(T model, string ext = "")
    {
        var directory = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (directory != null && !directory.GetDirectories("Saves").Any())
        {
            directory = directory.Parent;
        }

        var modelType = model.GetType().ToString().Split(".").Last();
        modelType += string.IsNullOrEmpty(ext) ? "" : $"_{ext}";
        
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