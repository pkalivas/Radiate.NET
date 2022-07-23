using System.IO.Compression;
using Newtonsoft.Json;

namespace Radiate.Data.Utils;

public static class Utilities
{
    public static async Task<T> UnzipGZAndLoad<T>(string filePath)
    {
        await using var fileStream = File.OpenRead(filePath);
        await using var zippedStream = new GZipStream(fileStream, CompressionMode.Decompress);
        
        using var jsonReader = new JsonTextReader(new StreamReader(zippedStream));

        return JsonSerializer.CreateDefault().Deserialize<T>(jsonReader);
    }
    
    public static async Task<T> UnzipGZAndLoad<T>(Stream fileStream)
    {
        await using var zippedStream = new GZipStream(fileStream, CompressionMode.Decompress);
        
        using var jsonReader = new JsonTextReader(new StreamReader(zippedStream));

        return JsonSerializer.CreateDefault().Deserialize<T>(jsonReader);
    }

    public static (List<float[]> inputs, List<float[]> targets) ReadCsv(string contents)
    {
        var features = new List<float[]>();
        var labels = new List<float[]>();
        foreach (var row in contents.Split("\n").Skip(1))
        {
            var columns = row
                .Split(",")
                .Skip(1)
                .Select(Convert.ToSingle)
                .ToList();
            
            features.Add(columns.Take(columns.Count - 1).ToArray());
            labels.Add(columns.Skip(columns.Count - 1).ToArray());
        }
        
        return (features, labels);
    }
    
    public static async Task<(List<float[]> inputs, List<float[]> targets)> LoadCsv(string filePath)
    {
        var contents = await File.ReadAllTextAsync(filePath);

        var features = new List<float[]>();
        var labels = new List<float[]>();
        foreach (var row in contents.Split("\n").Skip(1))
        {
            var columns = row
                .Split(",")
                .Skip(1)
                .Select(Convert.ToSingle)
                .ToList();
            
            features.Add(columns.Take(columns.Count - 1).ToArray());
            labels.Add(columns.Skip(columns.Count - 1).ToArray());
        }
        
        return (features, labels);
    }
}