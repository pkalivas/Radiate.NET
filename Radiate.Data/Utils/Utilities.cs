using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace Radiate.Data.Utils
{
    public static class Utilities
    {
        public static async Task<(List<List<float>>, List<List<float>>)> ReadBostonHousing()
        {
            var fileName = $"{Environment.CurrentDirectory}\\DataSets\\BostonHousing\\Boston.csv";
            var contents = await File.ReadAllTextAsync(fileName);

            var features = new List<List<float>>();
            var labels = new List<List<float>>();
            foreach (var row in contents.Split("\n").Skip(1))
            {
                var columns = row
                    .Split(",")
                    .Skip(1)
                    .Select(Convert.ToSingle)
                    .ToList();

                features.Add(columns.Take(columns.Count - 1).ToList());
                labels.Add(columns.Skip(columns.Count - 1).ToList());
            }

            return (features, labels);
        }
        
        public static async Task<T> UnzipGZAndLoad<T>(string filePath)
        {
            await using var fileStream = File.OpenRead(filePath);
            await using var zippedStream = new GZipStream(fileStream, CompressionMode.Decompress);
            
            using var jsonReader = new JsonTextReader(new StreamReader(zippedStream));

            return JsonSerializer.CreateDefault().Deserialize<T>(jsonReader);
        }
    }
}